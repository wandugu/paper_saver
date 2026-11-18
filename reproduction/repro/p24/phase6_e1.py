"""Ledgered phase 6 E1 replay and post hoc comparisons (CPU only)."""

import gzip
import json
import math
import os
import struct
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch
import transformers

from repro.common.paths import require_env
from repro.common.prereg import require_fixed_prereg

from .phase3_replay import auroc, json_read
from .phase4_replay import find_run
from .phase5_analysis import align, paired, population, study_sources, subset
from .phase6_statistics import comparison, counts_by_sign, interval, paired_f1, post_gain, seed_t, signs
from .protocol import digest, read_records, task_f1


PREREG = Path(__file__).with_name("prereg_p24_phase6_e1.yaml")


def exact(actual, expected, label):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: keys differ: {sorted(actual)} vs {sorted(expected)}")
        for key in expected:
            exact(actual[key], expected[key], f"{label}.{key}")
    elif type(actual) is not type(expected) or (struct.pack(">d", actual) != struct.pack(">d", expected)
            if isinstance(expected, float) else actual != expected):
        raise ValueError(f"{label}: replay={actual!r}; phase5={expected!r}")


def as_display(value):
    return f"{value:.5f}"


def a6_run(root, seed, method):
    return find_run(root, seed, experiment="phase5_training", tag_key="study", tag_value="A6",
                    extra_tags={"method": method})[0]


def a6_replay(prereg, manifest, record):
    root = Path(prereg["phase5_root"])
    c3, _ = find_run(root, 42, experiment="phase5_checkpoint", tag_key="checkpoint", tag_value="C3")
    reference = json_read(c3 / "metrics.json")["seeds"]
    record("reference", "C3", c3 / "metrics.json")
    data = {"dev": {}, "test": {}}
    for offset, seed in enumerate(prereg["seeds"]):
        text, vision = (a6_run(root, seed, method) for method in ("text_only", "always_vision"))
        for split in data:
            paths = [text / f"raw/{split}.jsonl", vision / f"raw/{split}.jsonl"]
            text_rows, vision_rows = map(read_records, paths)
            for path in paths:
                record("prediction", f"A6/{split}/{seed}", path)
            pair = comparison(vision_rows, text_rows, manifest)
            if len(pair[0].ids) != prereg["E1a"]["units"][split]:
                raise ValueError(f"A6 {split}/{seed}: wrong sample size")
            text_f1 = task_f1(text_rows, "mre")
            vision_f1 = task_f1(vision_rows, "mre")
            metric = "development_f1" if split == "dev" else "test_f1"
            for directory, actual in ((text, text_f1), (vision, vision_f1)):
                record("reference", f"A6/{split}/{seed}", directory / "metrics.json")
                exact(actual, json_read(directory / "metrics.json")[metric], f"{directory.name}/{metric}")
            original = None
            if split == "dev":
                original = paired(*pair, seed)
                observed = {**original, "text_run_id": text.name, "vision_run_id": vision.name,
                            "text_dev_f1": text_f1, "vision_dev_f1": vision_f1}
                exact(observed, reference[f"s{seed}"], f"A6 dev/{seed}")
                for observed_value, target in zip((original["f1_delta"], original["ci_low"], original["ci_high"]),
                                                  prereg["E1a"]["dev_display"][seed]):
                    exact(as_display(observed_value), target, f"A6 dev display/{seed}")
                exact(original["n_posts"], prereg["E1a"]["dev_posts"], f"A6 dev posts/{seed}")
            else:
                exact(as_display(text_f1), prereg["E1a"]["test_display_text"][offset], f"A6 test text/{seed}")
                exact(as_display(vision_f1), prereg["E1a"]["test_display_vision"][offset], f"A6 test vision/{seed}")
            data[split][seed] = (pair, original, text_f1, vision_f1, text.name, vision.name)
    return data


def auc_replay(prereg, manifest, record):
    root = Path(prereg["phase5_root"])
    sources = json_read(prereg["sources"])
    if "P24_PHASE1_RUNS_ROOT" in os.environ:
        sources["phase1_root"] = require_env("P24_PHASE1_RUNS_ROOT")
    result = {}
    for study in prereg["E1b"]["studies"]:
        result[study] = {}
        for offset, seed in enumerate(prereg["seeds"]):
            run, _ = find_run(root, seed, experiment="phase5_analysis", tag_key="study", tag_value=study,
                              extra_tags={"analysis_revision": prereg["E1b"]["analysis_revision"]})
            config = json_read(run / "config.json")
            config["phase5_root"] = str(root)
            _, trained, _, identity, anchors, branches, text, _ = study_sources(config, seed, sources=sources)
            extract, _ = find_run(root, seed, experiment="phase5_anchor", tag_key="scope", tag_value="full")
            extension, _ = find_run(root, seed, experiment="phase5_extension", tag_key="study", tag_value=study)
            for path in (trained / "raw/calibration.jsonl", extension / "raw/calibration_extension.jsonl",
                         text / "raw/calibration.jsonl", extract / "raw/calibration_extension.jsonl"):
                record("prediction", f"{study}/calibration/{seed}", path)
            record("reference", f"{study}/calibration/{seed}", run / "metrics.json")
            expected = json_read(run / "metrics.json")["auroc"]
            ids = identity["split_ids"]["calibration"]
            branch_rows, missing_branch = subset(branches["saver"]["calibration"], ids)
            anchor_rows, missing_anchor = subset(anchors["calibration"], ids)
            if missing_branch or missing_anchor:
                raise ValueError(f"{study}/{seed}: incomplete calibration")
            scores, summaries = {}, {}
            gains = None
            for scorer, display in (("saver", "gate_display"), ("uncertainty", "uncertainty_display")):
                pop = population(align(branch_rows, anchor_rows, scorer=scorer), manifest)
                current = post_gain(pop, pop.vision, pop.text)
                if gains is not None and not np.array_equal(current, gains):
                    raise ValueError("AUROC labels differ by scorer")
                gains = current
                score = np.bincount(pop.groups, weights=pop.gate) / pop.counts
                point = auroc(score, gains > 0)
                summary = counts_by_sign(gains)
                target = expected[scorer]["post"]
                for key, actual in (("auroc", point), ("positive", summary["fix"]), ("negative", summary["break"]),
                                    ("zero", summary["zero"]), ("n", len(gains))):
                    exact(actual, target[key], f"{study}/{seed}/{scorer}/{key}")
                exact(len(gains), prereg["E1b"]["posts_per_seed"], f"{study}/{seed}/posts")
                exact(as_display(point), prereg["E1b"][display][study][offset], f"{study}/{seed}/{scorer}/display")
                scores[scorer], summaries[scorer] = score, point
                if scorer == "saver":
                    source_population = pop
                    unit_gate = pop.gate
                else:
                    unit_uncertainty = pop.gate
            if study == "A1":
                exact([summary["fix"], summary["break"]], prereg["E1c"]["A1_fix_break"][offset], f"A1/{seed}/signs")
            result[study][seed] = (source_population, gains, scores, summaries, summary, run.name,
                                    unit_gate, unit_uncertainty)
    return result


def auc_difference(gains, scores, seed, bootstrap):
    labels = gains > 0
    generator = np.random.default_rng(bootstrap["base"] + seed)
    deltas = []
    for _ in range(bootstrap["draws"]):
        draw = generator.integers(0, len(labels), len(labels))
        sample = labels[draw]
        if sample.any() and not sample.all():
            deltas.append(auroc(scores["uncertainty"][draw], sample) - auroc(scores["saver"][draw], sample))
    return {**interval(deltas), "invalid_draws": bootstrap["draws"] - len(deltas)}


def sign_summary(rows):
    return signs(*(sum(row[key] for row in rows.values()) for key in ("fix", "break", "zero")))


def main(ctx, config):
    prereg = require_fixed_prereg(ctx, PREREG)
    if config != {"preregistered": prereg["id"]}:
        raise ValueError("Use the committed E1 preregistration and fixed configuration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise RuntimeError("E1 is CPU only")
    if (torch.__version__, transformers.__version__) != (prereg["runtime"]["torch"], prereg["runtime"]["transformers"]):
        raise RuntimeError("E1 needs the pinned phase5 runtime")
    torch.set_num_threads(prereg["runtime"]["torch_threads"])
    ctx.tags.update(stage="phase6", study="E1", post_hoc=True, device="cpu")
    sources = {}

    def record(kind, label, path):
        key = str(path)
        if key not in sources:
            sources[key] = {"kind": kind, "used_by": [], "sha256": digest(path), "bytes": path.stat().st_size}
        sources[key]["used_by"].append(label)

    try:
        ledger = Path(prereg["phase5_root"]) / "ledger.jsonl"
        manifest_path = Path(json_read(prereg["sources"])["manifest"])
        for path, key in ((ledger, "phase5_ledger_sha256"), (manifest_path, "manifest_sha256")):
            record("input", key, path)
            exact(sources[str(path)]["sha256"], prereg[key], key)
        manifest = read_records(manifest_path)
        a6 = a6_replay(prereg, manifest, record)
        auc = auc_replay(prereg, manifest, record)
    except Exception as error:
        ctx.set_metrics({"exact_replay": False})
        ctx.save_json("replay_failure.json", {"reason": str(error), "sources": sources})
        raise

    ctx.save_json("source_identity.json", sources)
    metrics = {"exact_replay": True, "exact_replay_numeric": 1, "E1a": {}, "E1b": {}, "E1c": {}}
    with gzip.GzipFile(filename="", mode="wb", fileobj=(ctx.run_dir / "E1_examples.jsonl.gz").open("xb"), mtime=0) as evidence:
        def write(row):
            evidence.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())

        for split, seeds in a6.items():
            observed, discordance = {}, {}
            for seed, (pair, original, text_f1, vision_f1, text_id, vision_id) in seeds.items():
                settings = prereg["E1a"]["test_seed_bootstrap"]
                base = 20261113 if split == "dev" else settings["base"]
                estimate = paired_f1([pair], settings["draws"], base + seed)
                if original is not None:
                    for key, old_key in (("point", "f1_delta"), ("low", "ci_low"), ("high", "ci_high")):
                        exact(estimate[key], original[old_key], f"A6 rebuilt bootstrap/{seed}/{key}")
                observed[str(seed)] = {**estimate, "text_f1": text_f1, "vision_f1": vision_f1,
                                       "text_run_id": text_id, "vision_run_id": vision_id}
                pop, vision, text = pair
                discordance[str(seed)] = counts_by_sign(post_gain(pop, vision, text))
                for unit_id, group, gold, v_pred, t_pred in zip(pop.ids, pop.groups, pop.gold, vision, text):
                    write({"study": "A6", "split": split, "seed": seed, "id": unit_id, "post": pop.post_ids[group],
                           "gold": int(gold), "vision_prediction": int(v_pred), "text_prediction": int(t_pred),
                           "vision_run_id": vision_id, "text_run_id": text_id})
            pool = prereg["E1a"]["pooled_bootstrap"]
            metrics["E1a"][split] = {"seeds": observed,
                                      "pooled": paired_f1([row[0] for row in seeds.values()], pool["draws"], pool["random_seeds"][split]),
                                      "seed_t": seed_t([row["point"] for row in observed.values()])}
            metrics["E1c"][f"A6_{split}"] = {**discordance, "pooled": sign_summary(discordance)}
        for study, seeds in auc.items():
            observed, discordance = {}, {}
            for seed, (pop, gains, scores, points, counts, run_id, unit_gate, unit_uncertainty) in seeds.items():
                observed[str(seed)] = {**points, "point": points["uncertainty"] - points["saver"], "posts": len(gains),
                                       "source_run_id": run_id, **auc_difference(gains, scores, seed, prereg["E1b"]["bootstrap"])}
                discordance[str(seed)] = counts
                for idx, unit_id in enumerate(pop.ids):
                    group = pop.groups[idx]
                    write({"study": study, "split": "calibration", "seed": seed, "id": unit_id,
                           "post": pop.post_ids[group], "gold": int(pop.gold[idx]), "text_prediction": int(pop.text[idx]),
                           "vision_prediction": int(pop.vision[idx]), "gate": float(unit_gate[idx]),
                           "uncertainty": float(unit_uncertainty[idx]), "post_gain": float(gains[group]), "source_run_id": run_id})
            metrics["E1b"][study] = {**observed, "mean": {key: float(np.mean([row[key] for row in observed.values()]))
                                                         for key in ("point", "saver", "uncertainty")}}
            metrics["E1c"][study] = {**discordance, "pooled": sign_summary(discordance)}
    ctx.set_metrics(metrics)


def calculate_epsilon(run, *, prereg_path=None):
    """Calculate the receipt using the registration file associated with the run."""
    run = Path(run)
    if not (run / "_SUCCESS").exists():
        raise ValueError("E1 must finish successfully before calculating epsilon")
    metrics = json_read(run / "metrics.json")
    exact(metrics["exact_replay"], True, "E1 bitwise replay")
    receipt = run / "preregistration.json"
    if receipt.is_file():
        prereg_sha256 = json_read(receipt)["sha256"]
        if prereg_path is not None and digest(prereg_path) != prereg_sha256:
            raise ValueError("Provided preregistration does not match the E1 run receipt")
    elif prereg_path is not None:
        prereg_sha256 = digest(prereg_path)
    else:
        raise ValueError("Historical E1 runs require the original preregistration file")
    se = metrics["E1a"]["dev"]["pooled"]["se"]
    raw = 2 * 1.645 * se
    ceiling_steps = math.ceil(raw / .005)
    rounded = ceiling_steps * .005
    epsilon = min(.05, max(.02, rounded))
    probability = NormalDist().cdf(epsilon / se - 1.645)
    return {"e1_run_id": run.name, "e1_metrics_sha256": digest(run / "metrics.json"),
            "e1_prereg_sha256": prereg_sha256, "SE_pool_dev": se, "epsilon_raw": raw,
            "ceil_steps": ceiling_steps, "grid_step": .005, "rounded": rounded,
            "lower_bound": .02, "upper_bound": .05, "upper_clipped": rounded > .05,
            "epsilon": epsilon, "normal_critical": 1.645, "null_z": epsilon / se - 1.645,
            "null_bounded_probability_approx": probability, "inconclusive_probability_approx": 1 - probability,
            "scope": "Single-bound normal approximation at true effect zero and the E1 dev SE; not the joint dev/test probability."}
