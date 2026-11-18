import copy
import json
import math
import random
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from repro.p24 import phase2_training as training
from repro.p24.model import SaverCore
from repro.p24.protocol import digest, split_records


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: pytest.fail("No CUDA in CPU tests"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: pytest.fail("No CUDA in CPU tests"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(original_threads)


MODEL_OPTIONS = dict(text_dim=3, vision_dim=4, hidden_dim=8, query_dim=4,
                     width_dim=2, num_heads=1, dropout=0.0)


def make_model(router="saver", **kwargs):
    return training.Phase2Saver(3, router=router, **dict(MODEL_OPTIONS, **kwargs))


def inputs(images=2):
    units = {"head": torch.randn(2, 9), "tail": torch.randn(2, 9),
             "head_width": torch.ones(2, dtype=torch.long),
             "tail_width": torch.ones(2, dtype=torch.long)}
    global_features = torch.randn(images, 4)
    details = {index: torch.randn(5, 4) for index in range(images)}
    return units, global_features, details


def item(identity, split="train", images=None):
    return {"id": identity, "split": split, "tokens": ["Ada", identity, "Bob"],
            "head": [0, 1], "tail": [2, 3], "label": "related",
            "images": ["image-a", "image-b"] if images is None else images}


class Encoders:
    device = torch.device("cpu")
    text_hidden = 3
    vision_hidden = 4

    def encode_text(self, record, task, return_units=False):
        return {"head": torch.arange(9, dtype=torch.float32).view(1, 9) / 9,
                "tail": torch.arange(9, dtype=torch.float32).view(1, 9).flip(-1) / 9,
                "head_width": torch.ones(1, dtype=torch.long),
                "tail_width": torch.ones(1, dtype=torch.long), "truncated": False}

    def encode_global(self, images):
        return torch.arange(len(images) * 4, dtype=torch.float32).reshape(len(images), 4) / 8

    def encode_detail(self, image, return_mask=False):
        features = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
        mask = torch.ones(5, dtype=torch.bool)
        return (features, mask) if return_mask else features


class Context:
    def __init__(self, root, seed=42, name="current"):
        self.runs_root = root
        self.run_dir = root / name
        self.run_dir.mkdir(parents=True)
        self.seed = seed
        self.tags, self.metrics, self.saved, self.rows = {}, {}, {}, {}
        self.events = []

    def save_json(self, name, payload):
        self.saved[name] = payload
        (self.run_dir / name).write_text(json.dumps(payload, allow_nan=False))

    def set_metrics(self, payload):
        self.metrics.update(payload)

    def log_event(self, **payload):
        self.events.append(payload)

    def write_rows(self, name, rows):
        self.rows.setdefault(name, []).extend(rows)
        directory = self.run_dir / "raw"
        directory.mkdir(exist_ok=True)
        with (directory / f"{name}.jsonl").open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")


def setup_run(tmp_path, monkeypatch, *, seed=42, name="current", **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = ([item(f"train-{index}") for index in range(19)]
               + [item(f"dev-{index}", "dev") for index in range(4)]
               + [item(f"test-{index}", "test") for index in range(2)])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    config = {"task": "mre", "method": "saver", "router": "saver", "teacher": "instantaneous",
              "manifest": str(manifest), "inventory": str(inventory), "text_model": "mock-text",
              "vision_model": "mock-vision", "cache": str(tmp_path / "features"), **overrides}
    model_class = training.Phase2Saver

    def model_factory(num_labels, **kwargs):
        return model_class(num_labels, **dict(kwargs, **MODEL_OPTIONS))

    monkeypatch.setattr(training, "Phase2Saver", model_factory)
    monkeypatch.setattr(training.phase1, "gpu_guard", lambda pilot: {"cpu_mock": True, "physical_gpu": 1})
    monkeypatch.setattr(training.phase1, "FrozenEncoders", lambda *args: Encoders())
    monkeypatch.setattr(training.phase1, "verify_resources", lambda *args: {"test_identity": "cpu"})
    return Context(tmp_path, seed=seed, name=name), config, records


def parent_run(ctx, config, records, *, seed=None, status="SUCCESS"):
    directory = ctx.runs_root / "P24" / "phase1_parent"
    directory.mkdir(parents=True)
    (directory / "_SUCCESS").write_text("done\n")
    (directory / "config.json").write_text(json.dumps(dict(config, method="saver", pilot=False)))
    mapping = training.phase1._mapping([record for record in records if record["split"] == "train"], "mre")
    splits = split_records(records, ctx.seed)
    (directory / "data_identity.json").write_text(json.dumps({"sha256": digest(config["manifest"]),
        "labels": mapping, "split_ids": {name: [record["id"] for record in values] for name, values in splits.items()}}))
    (directory / "best.json").write_text(json.dumps({"epoch": 2, "step": 100, "dev_f1": .2}))
    torch.save(SaverCore(len(mapping), **MODEL_OPTIONS).state_dict(), directory / "best.pt")
    (ctx.runs_root / "ledger.jsonl").write_text(json.dumps({"paper": "P24", "experiment": "experiment",
        "seed": ctx.seed if seed is None else seed, "status": status, "run_dir": "P24/phase1_parent",
        "run_id": "phase1_parent"}) + "\n")
    return directory, mapping, splits


@pytest.mark.parametrize("router", ["linear", "utility_mlp", "saver", "UtilityMLP_K1"])
def test_routers_use_exact_unmodified_gate_inputs_and_parameter_budget(router):
    torch.manual_seed(91)
    reference = make_model().eval()
    torch.manual_seed(91)
    model = make_model(router).eval()
    units, global_features, details = inputs()
    captured = []
    hook = model.gate_mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    result = model(units, global_features, details, labels=torch.tensor([1, 2]), force_details=True)
    hook.remove()
    projected = F.normalize(model.global_projection(global_features), dim=-1)
    best = projected[result["retrieved"][:, 0]]
    expected = torch.cat((result["query"], best, result["query"] * best,
                          (result["query"] - best).abs(), result["similarity_stats"]), -1)
    assert torch.equal(captured[0], expected)
    parameters = sum(parameter.numel() for parameter in model.gate_mlp.parameters())
    reference_parameters = sum(parameter.numel() for parameter in reference.gate_mlp.parameters())
    if router == "linear":
        assert isinstance(model.gate_mlp, nn.Linear)
        assert parameters == 4 * model.query_dim + 5 < reference_parameters
    else:
        assert parameters == reference_parameters
        reference_result = reference(units, global_features, details, force_details=True)
        assert torch.equal(result["gate"], reference_result["gate"])
        assert torch.equal(result["logits"], reference_result["logits"])
    for name, parameter in model.named_parameters():
        if not name.startswith("gate_mlp"):
            assert torch.equal(parameter, dict(reference.named_parameters())[name])


def test_instantaneous_losses_and_saturated_gate_match_phase1_exactly():
    torch.manual_seed(65)
    base = SaverCore(3, **MODEL_OPTIONS).eval()
    torch.manual_seed(65)
    model = make_model().eval()
    with torch.no_grad():
        base.gate_mlp[-1].bias.fill_(8)
        model.gate_mlp[-1].bias.fill_(8)
    units, global_features, details = inputs()
    kwargs = dict(labels=torch.tensor([1, 2]), training=True, step=750)
    expected = base(units, global_features, details, **kwargs)
    actual = model(units, global_features, details, **kwargs)
    for key in ("loss", "task_loss", "utility_loss", "budget_loss", "consistency_loss",
                "utility_targets", "logits", "gate", "eta"):
        assert torch.equal(actual[key], expected[key])
    assert bool((actual["gate"] > .999).all())


def test_teacher_recomputes_same_features_no_grad_and_overrides_only_utility():
    calls = []

    class Teacher(SaverCore):
        def forward(self, units, global_features, *args, **kwargs):
            calls.append((units, global_features, torch.is_grad_enabled(), self.training))
            return super().forward(units, global_features, *args, **kwargs)

        def compute_losses(self, output, labels, **kwargs):
            result = super().compute_losses(output, labels, **kwargs)
            result["text_losses"] = torch.tensor([1., 1.])
            result["vision_losses"] = torch.tensor([.8, 1.1])
            return result

    model = make_model()
    teacher = Teacher(3, **MODEL_OPTIONS)
    model.set_teacher(teacher, "frozen")
    model.train()
    units, global_features, details = inputs()
    detail_calls = []

    def provider(indices):
        detail_calls.extend(indices)
        return {index: details[index] for index in indices}

    output = model(units, global_features, detail_provider=provider,
                   labels=torch.tensor([1, 2]), training=True)
    assert len(calls) == 1
    assert calls[0][0] is units and calls[0][1] is global_features
    assert calls[0][2:] == (False, False)
    assert not teacher.training
    assert output["utility_targets"].tolist() == [1, 0]
    assert not output["utility_targets"].requires_grad
    assert output["utility_loss"] == F.binary_cross_entropy(output["gate"], torch.tensor([1., 0.]))
    expected = (output["task_loss"] + .25 * output["utility_loss"]
                + .05 * output["budget_loss"] + .05 * output["consistency_loss"])
    assert torch.equal(output["loss"], expected)
    output["loss"].backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not set(map(id, teacher.parameters())) & set(map(id, model.parameters()))
    assert not any("teacher" in key for key in model.state_dict())
    assert len(detail_calls) == len(set(detail_calls))
    assert model._teacher_output is None
    model.eval()
    model(units, global_features, details, training=False, labels=torch.tensor([1, 2]))
    assert len(calls) == 1


def test_teacher_exception_clears_loss_context():
    model = make_model()
    model.set_teacher(copy.deepcopy(model), "ema")
    units, global_features, details = inputs()
    with pytest.raises(ValueError, match="shape"):
        model(units, global_features, details, labels=torch.tensor([1]))
    assert model._teacher_output is None


@pytest.mark.parametrize("router", ["linear", "utility_mlp", "saver"])
def test_no_image_is_exact_fallback_with_detached_teacher(router):
    model = make_model(router)
    model.set_teacher(copy.deepcopy(model), "ema")
    units, global_features, _ = inputs(images=0)
    output = model(units, global_features, labels=torch.tensor([1, 2]), training=True)
    assert torch.equal(output["logits"], output["text_logits"])
    assert not output["active"].any()
    assert output["utility_loss"] == 0


def test_ema_is_parameterwise_stop_gradient_and_preserves_eval():
    student = make_model()
    teacher = copy.deepcopy(student)
    student.set_teacher(teacher, "ema")
    before = {name: parameter.detach().clone() for name, parameter in teacher.named_parameters()}
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(2)
    training.update_ema(teacher, student, .99)
    for name, parameter in teacher.named_parameters():
        expected = before[name] * .99 + dict(student.named_parameters())[name].detach() * .01
        assert torch.allclose(parameter, expected)
        assert not parameter.requires_grad
    assert not teacher.training
    with pytest.raises(ValueError):
        training.update_ema(teacher, student, 1)


def test_common_diagnostic_ids_are_independent_of_seed_fit_and_dev_shuffle():
    records = [item(f"dev-{index:04d}", "dev") for index in range(600)]
    expected = [record["id"] for record in training.diagnostic_records(records)]
    for seed in (42, 43, 44):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        assert [record["id"] for record in training.diagnostic_records(shuffled)] == expected
    assert len(expected) == 512
    assert expected != [record["id"] for record in split_records(records, 42)["dev"]]
    with pytest.raises(ValueError, match="unique"):
        training.diagnostic_records([records[0], records[0]])


@pytest.mark.parametrize("scores,labels,auc,brier", [
    ([.1, .9], [0, 1], 1., .01), ([.9, .1], [0, 1], 0., .81),
    ([.5, .5], [0, 1], .5, .25), ([.2, .4, .4, .8], [0, 1, 0, 1], .875, .15),
])
def test_utility_metrics_use_own_student_targets_with_tie_correct_auroc(scores, labels, auc, brier):
    rows = [{"gate": score, "student_target": label, "target": 1 - label, "eligible": True}
            for score, label in zip(scores, labels)]
    result = training.utility_metrics(rows)
    assert result["utility_auroc"] == pytest.approx(auc)
    assert result["utility_brier"] == pytest.approx(brier)


def test_single_class_and_empty_utility_auc_are_not_fabricated():
    assert training.utility_metrics([])["utility_auroc_reason"] == "empty_population"
    result = training.utility_metrics([{"gate": .5, "student_target": 0, "eligible": True}])
    assert result["utility_auroc"] is None and result["utility_brier"] == .25
    assert result["utility_auroc_reason"] == "single_class"
    with pytest.raises(RuntimeError):
        training.utility_metrics([{"gate": float("nan"), "student_target": 0, "eligible": True}])


def test_epoch_flip_rates_align_by_stable_unit_not_position():
    first = [{"unit_id": "a:0", "target": 0, "student_target": 1, "eligible": True},
             {"unit_id": "b:0", "target": 1, "student_target": 0, "eligible": True}]
    second = [dict(first[1], target=0), first[0]]
    result = training.flip_metrics([first, second])
    assert result["epoch_flip_rate"] == .5
    assert result["student_epoch_flip_rate"] == 0
    assert result["utility_transitions"] == 2
    with pytest.raises(ValueError, match="changed"):
        training.flip_metrics([first, first[:1]])


@pytest.mark.parametrize("overrides", [{"epochs": 9}, {"effective_batch": 8}, {"max_steps": 1},
    {"fit_cap": 32}, {"teacher": "ema", "router": "linear"}, {"checkpoint_selection": "test"},
    {"task": "mner"}, {"ema_decay": .9}, {"pilot": True, "pilot_diagnostic_cap": 0}])
def test_unregistered_configurations_fail_before_gpu_probe(tmp_path, monkeypatch, overrides):
    ctx = Context(tmp_path)
    monkeypatch.setattr(training.phase1, "gpu_guard", lambda *args: pytest.fail("GPU probe must not run"))
    with pytest.raises(ValueError):
        training.main(ctx, {"task": "mre", **overrides})


@pytest.mark.parametrize("router,teacher", [("linear", "instantaneous"), ("utility_mlp", "instantaneous"),
                                          ("saver", "instantaneous"), ("saver", "ema")])
def test_pilot_main_runs_small_cpu_smoke_without_test_access(tmp_path, monkeypatch, router, teacher):
    ctx, config, records = setup_run(tmp_path, monkeypatch, router=router, teacher=teacher,
                                     pilot=True, max_steps=1, pilot_epochs=1)
    calls = []
    original = training.phase1.evaluate

    def evaluate(*args, **kwargs):
        calls.append(kwargs.get("test", False))
        assert not kwargs.get("test")
        return original(*args, **kwargs)

    monkeypatch.setattr(training.phase1, "evaluate", evaluate)
    training.main(ctx, config)
    assert ctx.tags["stage"] == "pilot"
    assert ctx.metrics["pilot_pass"]
    assert ctx.metrics["optimizer_steps"] == 1
    assert ctx.metrics["test_evaluations"] == 0
    assert not (ctx.run_dir / "TEST_STARTED").exists()
    assert (ctx.run_dir / "epoch_00.pt").exists() and (ctx.run_dir / "best.pt").exists()
    assert ctx.saved["PILOT_ONLY.json"]["test_access"] is False
    assert len(ctx.rows["utility_epochs"]) == len(training.diagnostic_records(records, 16))
    assert ctx.saved["calibration.json"]["policy"] == "phase1_pair_forced_error_CP_not_post_harm"
    row = ctx.rows["utility_final"][0]
    assert set(row) >= {"id", "unit_id", "unit_index", "epoch", "seed", "router", "teacher", "target",
                        "student_target", "gate", "eligible", "gold", "text_loss", "vision_loss"}
    assert row["stage"] == "selected"
    json.dumps(ctx.metrics, allow_nan=False)


@pytest.mark.parametrize("selection", ["hard_dev", "soft_dev", "last_epoch"])
def test_formal_main_completes_all_epochs_retains_e6_schema_and_tests_once(tmp_path, monkeypatch, selection):
    ctx, config, records = setup_run(tmp_path, monkeypatch, teacher="ema", checkpoint_selection=selection)
    original = training.phase1.evaluate
    test_calls = []
    original_ema = training.update_ema
    ema_steps = []
    optimizer_steps = []
    original_step = torch.optim.AdamW.step

    def optimizer_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        optimizer_steps.append(len(optimizer_steps) + 1)
        return result

    def update(teacher, student, decay):
        assert len(optimizer_steps) == len(ema_steps) + 1
        ema_steps.append(len(optimizer_steps))
        return original_ema(teacher, student, decay)

    def evaluate(*args, **kwargs):
        if kwargs.get("test"):
            assert (ctx.run_dir / "METHOD_LOCK.json").exists()
            assert (ctx.run_dir / "TEST_STARTED").exists()
            test_calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", optimizer_step)
    monkeypatch.setattr(training, "update_ema", update)
    monkeypatch.setattr(training.phase1, "evaluate", evaluate)
    training.main(ctx, config)
    entries = ctx.saved["epochs.json"]["epochs"]
    assert len(entries) == 10 and all(entry["complete"] for entry in entries)
    assert [entry["epoch"] for entry in entries] == list(range(10))
    expected_steps = math.ceil(len(split_records(records, 42)["fit"]) / 16) * 10
    assert ctx.metrics["optimizer_steps"] == len(ema_steps) == expected_steps
    assert all((ctx.run_dir / entry["checkpoint"]).is_file() for entry in entries)
    assert all(set(entry) >= {"epoch", "step", "checkpoint", "soft_dev_f1", "hard_dev_f1"} for entry in entries)
    best = ctx.saved["best.json"]
    if selection == "last_epoch":
        assert best["epoch"] == 9
    else:
        key = "hard_dev_f1" if selection == "hard_dev" else "soft_dev_f1"
        expected = max(entries, key=lambda entry: entry[key])
        assert best["epoch"] == expected["epoch"]
    assert {"epoch", "step", "dev_f1"} <= best.keys()
    assert len(test_calls) == 1 and ctx.metrics["test_evaluations"] == 1
    assert len(ctx.rows["test"]) == len(split_records(records, 42)["test"])
    final_rows = ctx.rows["utility_final"]
    selected_rows = [row for row in ctx.rows["utility_epochs"] if row["epoch"] == best["epoch"]]
    assert [dict(row, stage="selected") for row in selected_rows] == final_rows
    assert len(ctx.rows["utility_epochs"]) == 10 * len(training.diagnostic_records(records))
    assert ctx.saved["data_identity.json"]["split_ids"] == {
        name: [record["id"] for record in values] for name, values in split_records(records, 42).items()}
    assert ctx.saved["METHOD_LOCK.json"]["threshold"] == ctx.saved["calibration.json"]["threshold"]
    assert ctx.metrics["calibration_threshold"] is None
    assert test_calls[0]["threshold"] == math.inf
    assert ctx.metrics["activation_rate"] == 0
    assert ctx.metrics["test_post_harm"] == 0
    assert ctx.metrics["gpu_hours"] > 0
    assert (ctx.run_dir / "final_eval.json").is_file()


def test_test_lock_prevents_retraining_or_replacement(tmp_path, monkeypatch):
    ctx = Context(tmp_path)
    (ctx.run_dir / "TEST_STARTED").write_text("locked")
    monkeypatch.setattr(training.phase1, "gpu_guard", lambda *args: pytest.fail("Must stop before GPU probe"))
    with pytest.raises(RuntimeError, match="single test"):
        training.main(ctx, {"task": "mre"})


def test_frozen_teacher_loads_exact_same_seed_parent_and_never_updates(tmp_path, monkeypatch):
    ctx, config, records = setup_run(tmp_path, monkeypatch, teacher="frozen", pilot=True,
                                     max_steps=1, pilot_epochs=1)
    directory, _, _ = parent_run(ctx, config, records)
    config["parent_run"] = "phase1_parent"
    monkeypatch.setattr(training, "update_ema", lambda *args: pytest.fail("Frozen teacher must not update"))
    training.main(ctx, config)
    metadata = ctx.saved["phase2_protocol.json"]["parent"]
    assert metadata["checkpoint_sha256"] == digest(directory / "best.pt")
    assert metadata["seed"] == 42
    assert ctx.metrics["teacher_parameters"] == ctx.metrics["total_parameters"]


def test_frozen_teacher_uses_phase1_ledger_when_phase2_has_different_runs_root(tmp_path, monkeypatch):
    ctx, config, records = setup_run(tmp_path / "phase2" / "runs", monkeypatch, teacher="frozen",
                                     pilot=True, max_steps=1, pilot_epochs=1,
                                     pilot_dev_cap=8, pilot_diagnostic_cap=8)
    phase1_ctx = Context(tmp_path / "phase1" / "runs")
    directory, _, _ = parent_run(phase1_ctx, config, records)
    config["parent_run"] = str(directory)
    assert not (ctx.runs_root / "ledger.jsonl").exists()
    training.main(ctx, config)
    metadata = ctx.saved["phase2_protocol.json"]["parent"]
    assert Path(metadata["ledger"]) == phase1_ctx.runs_root / "ledger.jsonl"
    assert metadata["checkpoint_sha256"] == digest(directory / "best.pt")
    assert ctx.metrics["pilot_pass"]


@pytest.mark.parametrize("failure", ["missing", "wrong_seed", "failed", "superseded", "split", "hash"])
def test_frozen_parent_validation_never_substitutes_checkpoint(tmp_path, monkeypatch, failure):
    ctx, config, records = setup_run(tmp_path, monkeypatch, teacher="frozen")
    if failure == "missing":
        with pytest.raises(RuntimeError, match="parent_run"):
            training._parent_teacher(ctx, config, {}, {})
        return
    directory, mapping, splits = parent_run(ctx, config, records,
        seed=43 if failure == "wrong_seed" else 42, status="FAILED" if failure == "failed" else "SUCCESS")
    config["parent_run"] = str(directory)
    if failure == "superseded":
        (directory / "SUPERSEDED.json").write_text("{}")
    elif failure == "split":
        splits = dict(splits, dev=splits["dev"][:1])
    elif failure == "hash":
        Path(config["manifest"]).write_text("changed\n")
    with pytest.raises(RuntimeError, match="BLOCKED"):
        training._parent_teacher(ctx, config, mapping, splits)


def test_post_harm_uses_normalized_positive_increment_after_post_aggregation():
    records = [item("a", "test"), item("b", "test"), item("c", "test")]
    records[1]["tokens"] = records[0]["tokens"]
    rows = [{"id": "a", "prediction": [0], "text_prediction": [1], "gold": [1]},
            {"id": "b", "prediction": [1], "text_prediction": [0], "gold": [1]},
            {"id": "c", "prediction": [0], "text_prediction": [1], "gold": [1]}]
    assert training._post_harm(rows, records) == .5
