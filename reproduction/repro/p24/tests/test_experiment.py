import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from repro.p24 import experiment, protocol, queue
from repro.p24.model import SaverCore


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        yield
    torch.set_num_threads(original_threads)


def record(identity="sample", split="train", images=None):
    return {"id": identity, "split": split, "tokens": ["Ada", "knows", "Bob"],
            "head": [0, 1], "tail": [2, 3], "label": "related",
            "images": ["image-a", "image-b", "image-c"] if images is None else images,
            "spans": [[0, 1, "PERSON"]]}


class MockEncoders:
    device = torch.device("cpu")
    text_hidden = 3
    vision_hidden = 4

    def __init__(self, value=1.0):
        self.value = value
        self.text_calls = []
        self.global_calls = []
        self.detail_calls = []

    def encode_text(self, item, task, return_units=False):
        self.text_calls.append((item["id"], task))
        count = 1 if task == "mre" else 20
        spans = torch.tensor([[index, index + 1] for index in range(count)])
        output = {"head": torch.full((count, 9), self.value),
                  "head_width": torch.ones(count, dtype=torch.long), "truncated": False}
        if task == "mre":
            output.update(tail=torch.full((count, 9), self.value + 1),
                          tail_width=torch.ones(count, dtype=torch.long))
        else:
            output["spans"] = spans
        return output

    def encode_global(self, images):
        self.global_calls.append(list(images))
        return torch.full((len(images), 4), self.value)

    def encode_detail(self, image, return_mask=False):
        self.detail_calls.append(image)
        patches = torch.full((5, 4), self.value)
        mask = torch.tensor([True, True, True, False, False])
        return (patches, mask) if return_mask else patches


def model_factory(num_labels=2, **kwargs):
    kwargs.update(text_dim=3, vision_dim=4, hidden_dim=8, query_dim=4,
                  width_dim=2, num_heads=1, dropout=0.0)
    model = SaverCore(num_labels, **kwargs)
    model.cuda = lambda: model
    return model


class MockContext:
    seed = 42

    def __init__(self, root):
        self.runs_root = root
        self.run_dir = root / "current"
        self.run_dir.mkdir()
        self.tags = {}
        self.metrics = {}
        self.saved = {}
        self.rows = {}
        self.events = []

    def save_json(self, name, payload):
        self.saved[name] = payload
        (self.run_dir / name).write_text(json.dumps(payload))

    def set_metrics(self, values):
        self.metrics.update(values)

    def log_event(self, **values):
        self.events.append(values)

    def write_rows(self, name, rows):
        self.rows.setdefault(name, []).extend(rows)


def setup_mock_run(tmp_path, monkeypatch, *, method="saver"):
    ctx = MockContext(tmp_path)
    encoders = MockEncoders()
    splits = {name: [record(name, "train" if name in ("fit", "calibration") else name)]
              for name in ("fit", "dev", "calibration", "test")}
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item) for group in splits.values() for item in group) + "\n")
    config = {"task": "mre", "method": method, "manifest": str(manifest),
              "inventory": str(inventory), "text_model": "mock-text",
              "vision_model": "mock-vision", "cache": str(tmp_path / "features")}
    monkeypatch.setattr(experiment, "FrozenEncoders", lambda *args: encoders)
    monkeypatch.setattr(experiment, "SaverCore", model_factory)
    monkeypatch.setattr(experiment, "gpu_guard", lambda pilot: {"cpu_mock": True})
    monkeypatch.setattr(experiment, "verify_resources",
                        lambda config, task: {"identity": "mock-resources", "task": task}, raising=False)
    monkeypatch.setattr(experiment, "split_records", lambda *args, **kwargs: splits)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    return ctx, config, splits, encoders


@pytest.mark.parametrize("method", ["text_only", "always_vision", "maxsim", "random", "saver", "mean", "topk2"])
def test_forward_methods_use_shared_cpu_store_and_correct_labels(tmp_path, method):
    encoders = MockEncoders()
    store = protocol.FeatureStore(tmp_path, encoders)
    item = record()
    output, units, labels = experiment._forward(
        model_factory().eval(), store, item, "mre", {"None": 0, "related": 1}, method,
        1000, threshold=0, random_rate=1, seed=42,
    )
    assert labels.tolist() == [1]
    assert output["logits"].shape == (1, 2)
    assert torch.isfinite(output["loss"])
    assert units["head"].device.type == "cpu"
    assert output["global_screen_calls"] == (0 if method == "text_only" else 3)
    assert len(encoders.detail_calls) == {"text_only": 0, "always_vision": 3, "topk2": 2}.get(method, 1)


def test_feature_store_reuses_detached_cpu_features_and_filters_detail_padding(tmp_path):
    encoders = MockEncoders()
    store = protocol.FeatureStore(tmp_path, encoders)
    first = store.text_global(record(), "mre")
    second = store.text_global(record(), "mre")
    assert torch.equal(first["global"], second["global"])
    assert len(encoders.text_calls) == len(encoders.global_calls) == 1
    first_detail = store.detail("image-a")
    second_detail = store.detail("image-a")
    assert first_detail.shape == (3, 4)
    assert torch.equal(first_detail, second_detail)
    assert encoders.detail_calls == ["image-a"]
    store.detail("image-a", cache=False)
    assert encoders.detail_calls == ["image-a", "image-a"]


def test_text_only_cache_can_be_upgraded_to_global_features(tmp_path):
    encoders = MockEncoders()
    store = protocol.FeatureStore(tmp_path, encoders)
    assert store.text_global(record(), "mre", visual=False)["global"] is None
    payload = store.text_global(record(), "mre", visual=True)
    assert payload["global"].shape == (3, 4)
    assert len(encoders.global_calls) == 1


def test_test_evaluation_retains_per_example_outputs_and_bypasses_detail_cache(tmp_path):
    encoders = MockEncoders()
    store = protocol.FeatureStore(tmp_path, encoders)
    model = model_factory().eval()
    item = record()
    experiment.evaluate(model, store, [item], "mre", {"None": 0, "related": 1}, "always_vision", 1000)
    calls = len(encoders.detail_calls)
    rows = experiment.evaluate(model, store, [item], "mre", {"None": 0, "related": 1},
                               "always_vision", 1000, test=True)
    assert len(encoders.detail_calls) == calls + 3
    assert len(rows) == 1 and rows[0]["id"] == item["id"]
    assert rows[0]["gold"] == [1]
    assert rows[0]["active"] == [True]
    assert rows[0]["detail_encodes"] == rows[0]["screen_encodes"] == 3
    assert set(rows[0]) >= {"prediction", "probabilities", "gate", "selected", "patchids",
                            "text_prediction", "vision_prediction"}


def test_calibration_audit_forces_vision_without_overriding_gate(tmp_path):
    encoders = MockEncoders()
    store = protocol.FeatureStore(tmp_path, encoders)
    rows = experiment.evaluate(model_factory().eval(), store, [record()], "mre",
                               {"None": 0, "related": 1}, "saver", 1000,
                               threshold=math.inf, audit=True)
    row = rows[0]
    assert row["active"] == [False]
    assert row["detail_encodes"] == 1
    assert row["forced_error"] == [row["vision_prediction"][0] != row["gold"][0]]
    assert len(row["harm"]) == 1


def test_mner_negative_sampling_never_drops_positives(tmp_path):
    store = protocol.FeatureStore(tmp_path, MockEncoders())
    output, units, labels = experiment._forward(model_factory(2, task="mner"), store, record(),
                                                "mner", {"None": 0, "PERSON": 1}, "text_only", 0,
                                                training=True)
    assert units["spans"].shape == (20, 2)
    assert labels[0] == 1
    assert (labels == 0).sum() == 3
    assert (labels == -100).sum() == 16
    assert torch.isfinite(output["loss"])


def test_cp_audit_uses_forced_error_and_excludes_invalid_images():
    rows = [{"gate": [0.9] * 200 + [0.99], "forced_error": [False] * 200 + [True],
             "gold": [1] * 201, "eligible": [True] * 200 + [False],
             "harm": [True] * 201}]
    calibration = experiment._calibrate(rows, "mre", 42)
    assert calibration["feasible"]
    assert calibration["n"] == calibration["raw_valid_units"] == 200
    assert calibration["k"] == 0
    assert calibration["threshold"] == 0


def test_mner_calibration_balances_explicit_class_population():
    rows = [{"gate": [0.9] * 220, "forced_error": [False] * 220,
             "gold": [0] * 200 + [1] * 20, "eligible": [True] * 220}]
    calibration = experiment._calibrate(rows, "mner", 42)
    assert calibration["raw_valid_units"] == 220
    assert calibration["n_total"] == 40
    assert calibration["threshold"] is None
    assert not calibration["feasible"]


def test_mock_formal_runner_locks_before_exactly_one_test_pass(tmp_path, monkeypatch):
    ctx, config, splits, encoders = setup_mock_run(tmp_path, monkeypatch)
    calls = []
    evaluate = experiment.evaluate

    def observed_evaluate(*args, **kwargs):
        if kwargs.get("test"):
            assert (ctx.run_dir / "METHOD_LOCK.json").is_file()
            assert (ctx.run_dir / "TEST_STARTED").is_file()
            assert args[2] is splits["test"]
            calls.append("test")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(experiment, "evaluate", observed_evaluate)
    experiment.run(ctx, config)
    assert calls == ["test"]
    assert len(ctx.rows["test"]) == 1
    assert ctx.metrics["test_samples"] == 1
    assert ctx.saved["METHOD_LOCK.json"]["test_evaluations_allowed"] == 1
    assert len(encoders.text_calls) == 4


def test_mock_pilot_runner_never_predicts_test_and_stops_after_twelve_steps(tmp_path, monkeypatch):
    ctx, config, splits, encoders = setup_mock_run(tmp_path, monkeypatch)
    experiment.run(ctx, config, pilot=True)
    assert ctx.metrics["optimizer_steps"] == 12
    assert ctx.metrics["pilot_pass"]
    assert "test" not in ctx.rows
    assert "METHOD_LOCK.json" not in ctx.saved
    assert not (ctx.run_dir / "TEST_STARTED").exists()
    assert not any(identity == "test" for identity, task in encoders.text_calls)


def test_queue_does_not_launch_occupied_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "gpu_status", lambda index: {"memory_mib": 500, "temperature": 30, "utilization": 0})

    def forbidden(*args, **kwargs):
        pytest.fail("Queue attempted to launch on an occupied GPU")

    monkeypatch.setattr(queue.subprocess, "Popen", forbidden)
    result = queue.execute_job({"name": "mock", "gpu": 2}, tmp_path, tmp_path,
                               datetime.now(timezone.utc) + timedelta(hours=1))
    assert result["status"] == "WAITING_FOR_IDLE"


def test_failed_profile_log_does_not_suppress_pending_profile(tmp_path):
    parent = tmp_path / "P24" / "parent"
    parent.mkdir(parents=True)
    (parent / "config.json").write_text(json.dumps({"task": "mre", "method": "saver"}))
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"paper": "P24", "experiment": "experiment",
        "status": "SUCCESS", "seed": 42, "run_dir": "P24/parent"}) + "\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "profile-mre-saver-s42.log").write_text("Traceback: interrupted profiling\n")
    plan = {"lanes": [[{"seed": 42, "task": "mre", "method": "saver"}]]}
    assert list(queue._profile_jobs(plan, tmp_path, logs))


def test_feature_cache_does_not_silently_reuse_other_encoder_outputs(tmp_path):
    first = protocol.FeatureStore(tmp_path, MockEncoders(value=1.0), identity="first")
    second = protocol.FeatureStore(tmp_path, MockEncoders(value=2.0), identity="second")
    first.text_global(record(), "mre")
    result = second.text_global(record(), "mre")
    assert result["units"]["head"].eq(2).all()


def test_pilot_does_not_read_heldout_test_labels(tmp_path, monkeypatch):
    ctx, config, splits, encoders = setup_mock_run(tmp_path, monkeypatch)

    class HeldoutRecord(dict):
        def __getitem__(self, key):
            if key in {"label", "spans"}:
                raise AssertionError("Pilot accessed held-out test labels")
            return super().__getitem__(key)

    records = [splits["fit"][0], splits["dev"][0], splits["calibration"][0],
               HeldoutRecord(splits["test"][0])]
    monkeypatch.setattr(experiment, "read_records", lambda path: records)
    experiment.run(ctx, config, pilot=True)


def test_random_run_keeps_calibration_counts_and_reported_coverage_consistent(tmp_path, monkeypatch):
    ctx, config, splits, encoders = setup_mock_run(tmp_path, monkeypatch, method="random")
    monkeypatch.setattr(experiment, "_random_rate", lambda *args: (0.35, "mock-saver"))
    experiment.run(ctx, config)
    calibration = ctx.saved["calibration.json"]
    assert calibration["coverage"] == calibration["n"] / calibration["n_total"]


def test_interrupted_test_preserves_completed_example_outputs(tmp_path, monkeypatch):
    ctx, config, splits, encoders = setup_mock_run(tmp_path, monkeypatch)
    splits["test"].append(record("test-failure", "test"))
    forward = experiment._forward

    def fail_second_test(model, store, item, *args, **kwargs):
        if item["id"] == "test-failure":
            raise RuntimeError("Injected single-pass evaluation interruption")
        return forward(model, store, item, *args, **kwargs)

    monkeypatch.setattr(experiment, "_forward", fail_second_test)
    with pytest.raises(RuntimeError, match="Injected single-pass"):
        experiment.run(ctx, config)
    assert (ctx.run_dir / "TEST_STARTED").exists()
    assert [row["id"] for row in ctx.rows.get("test", [])] == ["test"]
