"""Required phase6 CPU stop-loss tests; run in file order with pytest -x."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from repro.common.paths import expand_config_paths
from repro.p24.encoders import FrozenEncoders
from repro.p24.experiment import verify_resources
from repro.p24.phase6_e1 import a6_run
from repro.p24.phase6_e2 import BatchedModernBERT, FineTune, SEEDS, batches, data_for, resources, runtime, set_seed
from repro.p24.phase6_readonly import CacheMiss, ReadOnlyStore
from repro.p24.protocol import digest, task_f1


VISUAL_MODULES = {"query_projection", "global_projection", "patch_projection", "patch_attention",
                  "evidence_projection", "fusion_mlp", "consistency_query", "consistency_evidence"}


@pytest.fixture(scope="module")
def identity():
    runtime(cpu=True)
    return verify_resources(resources(), "mre")


@pytest.fixture(scope="module")
def store(identity):
    return ReadOnlyStore(resources()["cache"], SimpleNamespace(device="cpu", span_features=FrozenEncoders.span_features), identity)


def file_inventory(root):
    files = {str(path.relative_to(root)): path.stat().st_size for path in root.rglob("*") if path.is_file()}
    return files, {"files": len(files), "bytes": sum(files.values())}


def test_01_all_visual_features_read_only(store, record_property):
    root = Path(resources()["cache"])
    before_files, before = file_inventory(root)
    checks = []
    try:
        for seed in SEEDS:
            splits, _ = data_for(seed)
            for split, records in splits.items():
                attachments = 0
                for row in records:
                    try:
                        value = store.text_global(row, "mre", visual=True)["global"]
                        assert value.shape == (len(row["images"]), 768)
                    except CacheMiss:
                        pass
                    for image in row["images"]:
                        try:
                            detail = store.detail(image)
                            assert detail.ndim == 2 and detail.shape[1] == 768 and 0 < detail.shape[0] <= 256
                        except CacheMiss:
                            pass
                        attachments += 1
                checks.append({"seed": seed, "split": split, "units": len(records), "detail_reads": attachments})
    finally:
        after_files, after = file_inventory(root)
        record_property("cache", json.dumps({"before": before, "after": after, "checks": checks}, sort_keys=True))
        record_property("misses", json.dumps(store.misses, sort_keys=True))
        assert before_files == after_files, (before, after)
    assert not store.misses, store.misses


def sample_dev():
    splits, _ = data_for(42)
    return random.Random(20261118).sample(sorted(splits["dev"], key=lambda row: row["id"]), 64)


def cached_comparison(store, records, encoder, *, mixed=False, absolute=None):
    from repro.p24.phase6_e2 import autocast

    encoder.eval()
    legacy = SimpleNamespace(tokenizer=encoder.tokenizer, max_length=128)
    legacy._tokenize = FrozenEncoders._tokenize.__get__(legacy)
    measurements = []
    with torch.inference_mode(), autocast(mixed):
        for batch in batches(records, batch_size=1 if absolute is not None else 16):
            for row, found in zip(batch, encoder(batch)):
                payload = store.text_global(row, "mre", visual=True)
                expected = payload["units"]
                words = FrozenEncoders._words(row)
                spans = [FrozenEncoders._entity_span(row, name, len(words)) for name in ("head", "tail")]
                legacy_inputs, start, end = FrozenEncoders._aligned_inputs(legacy, words, spans)
                cached = expected["word_states"].float().cpu()
                actual = found["word_states"].float().cpu()
                assert tuple(actual.shape) == tuple(cached.shape)
                structure = (found["token_count"] == legacy_inputs["input_ids"].shape[1]
                             and [start, end] == expected["word_offsets"].tolist()
                             and all(torch.equal(found[field].cpu(), expected[field])
                                     for field in ("word_offsets", "head_span", "tail_span")))
                cosine = float(F.cosine_similarity(actual, cached, dim=-1).min())
                max_abs = float((actual - cached).abs().max())
                scale = float(cached.abs().max())
                limit = absolute if absolute is not None else .01 * scale
                measurements.append({"id": row["id"], "min_word_cosine": cosine, "max_abs": max_abs,
                                     "max_cached_abs": scale, "limit": limit, "alignment": structure,
                                     "cached_dtype": str(expected["word_states"].dtype),
                                     "passed": structure and max_abs <= limit and (absolute is not None or cosine >= .999)})
    return measurements


def test_02_cpu_word_states(store, record_property):
    encoder = BatchedModernBERT("cpu")
    measurements = cached_comparison(store, sample_dev(), encoder)
    record_property("word_states", json.dumps(measurements, sort_keys=True))
    failed = [row for row in measurements if not row["passed"]]
    assert not failed, failed


def test_03_gradient_parameter_difference(store, record_property):
    splits, mapping = data_for(42)
    batch = splits["fit"][:16]
    sets = {}
    for method in ("text_only", "always_vision"):
        set_seed(42, "cpu")
        encoder = BatchedModernBERT("cpu")
        model = FineTune(encoder, mapping, 42, store)
        losses = [output["loss"] for output, _ in model(batch, method, training=True)]
        torch.stack(losses).mean().backward()
        sets[method] = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        expected_encoder = {name for name, _ in model.named_parameters() if name.startswith("encoder.")}
        assert expected_encoder and expected_encoder <= sets[method]
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        del model, encoder, losses
    extra = sets["always_vision"] - sets["text_only"]
    record_property("grad_non_none", json.dumps({key: sorted(value) for key, value in sets.items()}))
    record_property("vision_only_gradients", json.dumps(sorted(extra)))
    assert sets["text_only"] < sets["always_vision"]
    assert all(name.startswith("head.") and name.split(".")[1] in VISUAL_MODULES for name in extra)


def test_04_pinned_split_ids(identity, record_property):
    pinned = Path(expand_config_paths("${P24_RUN_ROOT}/p24_phase5_pinned"))
    checked = []
    for seed in SEEDS:
        splits, mapping = data_for(seed)
        expected = {"sha256": digest(resources()["manifest"]), "labels": mapping,
                    "split_ids": {name: [row["id"] for row in rows] for name, rows in splits.items()}}
        for method in ("text_only", "always_vision"):
            run = a6_run(pinned, seed, method)
            assert json.loads((run / "data_identity.json").read_text()) == expected
            checked.append(run.name)
    record_property("pinned_run_ids", json.dumps(checked))


def test_05_f1_excludes_none():
    rows = [{"prediction": [predicted], "gold": [gold]} for predicted, gold in [(0, 0), (1, 1), (1, 0), (0, 2)]]
    assert task_f1(rows, "mre") == .5
    assert task_f1([{"prediction": [0], "gold": [0]}], "mre") == 0.
