"""Phase5 anchor-relative certificates and diagnostics from frozen prediction files."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .phase3_replay import Population, calibration_populations, json_read, paired, post_id
from .phase4_replay import CAPS, Phase4Population, calibrate_dual, choose_dual, policies
from .protocol import read_records, task_f1


def _ledger_rows(root):
    return read_records(Path(root) / "ledger.jsonl")


def _find_success(root, seed, *, experiment, tags):
    matches = [row for row in _ledger_rows(root) if row.get("paper") == "P24" and row.get("experiment") == experiment
               and row.get("seed") == seed and row.get("status") == "SUCCESS"
               and all(row.get("tags", {}).get(key) == value for key, value in tags.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one successful {experiment} seed={seed} tags={tags}, found {len(matches)}")
    return Path(root) / matches[0]["run_dir"], matches[0]


def _rows(path):
    return {row["id"]: row for row in read_records(path)}


def _phase1_run(root, seed, method):
    return _find_success(root, seed, experiment="experiment", tags={"method": method, "stage": "formal"})[0]


def _post_groups(rows, manifest):
    by_id = {row["id"]: row for row in manifest}
    ids = list(rows)
    groups = {}
    for index, key in enumerate(ids):
        groups.setdefault(post_id(by_id[key]), []).append(index)
    return list(groups.values())


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    return float((rankdata(scores)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _bootstrap_auroc(scores, labels, groups, seed):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    groups = list(groups)
    point = _auroc(scores, labels)
    if point is None:
        return {"point": None, "ci_low": None, "ci_high": None, "valid": 0}
    generator = np.random.default_rng(20261114 + seed)
    values = []
    for _ in range(2000):
        sample = generator.integers(0, len(groups), len(groups))
        indices = np.concatenate([np.asarray(groups[index], dtype=int) for index in sample])
        value = _auroc(scores[indices], labels[indices])
        if value is not None:
            values.append(value)
    return {"point": point, "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975)),
            "valid": len(values), "replicates": 2000}


def _with_scorer(rows, scorer):
    result = {}
    for key, row in rows.items():
        copied = dict(row)
        if scorer == "saver":
            score = row["gate"][0]
        elif scorer == "text_uncertainty_1_minus_max_p":
            probabilities = row.get("text_probabilities") or row.get("probabilities")
            score = 1.0 - max(probabilities[0])
        else:
            score = row["gate"][0]
        copied["gate"] = [float(score)]
        result[key] = copied
    return result


def _anchor_rows(rows, anchor):
    result = {}
    for key, row in rows.items():
        copied = dict(row)
        copied["text_prediction"] = [int(anchor[key]["prediction"][0] if "prediction" in anchor[key] else anchor[key]["text_prediction"][0])]
        result[key] = copied
    return result


def _population(rows, manifest):
    ordered = [rows[key] for key in sorted(rows)]
    return Phase4Population(ordered, manifest, None)


def _select_for_fraction(rows, manifest, fraction_ids, dev_rows, test_rows, seed):
    calibration_rows = {key: rows[key] for key in fraction_ids if key in rows}
    if len(calibration_rows) != len(fraction_ids):
        return {"missing": sorted(set(fraction_ids) - set(calibration_rows))}
    calibration = _population(calibration_rows, manifest)
    development = _population(dev_rows, manifest)
    test = _population(test_rows, manifest)
    output = {}
    for scorer in ("saver", "maxsim", "text_uncertainty_1_minus_max_p"):
        scored_cal = _with_scorer(calibration_rows, scorer)
        scored_dev = _with_scorer(dev_rows, scorer)
        scored_test = _with_scorer(test_rows, scorer)
        cal = _population(scored_cal, manifest)
        dev = _population(scored_dev, manifest)
        test_pop = _population(scored_test, manifest)
        output[scorer] = {}
        for cap in CAPS:
            curve = calibrate_dual(cal, policies(), cap)
            selected, fallback, feasible = choose_dual(curve, dev)
            dev_metric = dev.evaluate(selected)
            test_metric = test_pop.evaluate(selected)
            output[scorer][f"c{int(cap * 100):02d}"] = {
                "threshold": selected["threshold"], "harm_ucb": selected["harm_ucb"], "cost_ucb": selected["cost_ucb"],
                "calibration_harm": selected["harm"], "calibration_cost": selected["cost"],
                "dev": dev_metric, "test": test_metric, "fallback": int(fallback),
                "feasible_nonempty": int(bool(feasible)), "feasible_policies": len(feasible),
                "discriminative": {"calibration_harm": selected["harm"], "dev_f1": dev_metric["f1"],
                                   "d_post": dev_metric["detail"]},
            }
    return output


def _diagnostics(rows, manifest, seed):
    population = _population(rows, manifest)
    forced = np.asarray([row["vision_prediction"][0] for row in rows.values()])
    anchor = np.asarray([row["text_prediction"][0] for row in rows.values()])
    gold = np.asarray([row["gold"][0] for row in rows.values()])
    scorer = np.asarray([row["gate"][0] for row in rows.values()])
    groups = _post_groups(rows, manifest)
    unit_labels = (forced == gold) & (anchor != gold)
    anchor_unit = anchor == gold
    post_scores = np.asarray([float(np.mean(scorer[index])) for index in groups])
    post_labels = np.asarray([float(np.mean(forced[index] == gold[index]) - np.mean(anchor[index] == gold[index])) > 0 for index in groups])
    post_groups = [[index] for index in range(len(groups))]
    unit_groups = groups
    confidence = np.asarray([max(row.get("text_probabilities", [[0.]])[0]) for row in rows.values()])
    order = np.argsort(confidence)
    quintiles = []
    for bucket in np.array_split(order, 5):
        if len(bucket):
            fix = float(np.mean(unit_labels[bucket] & ~anchor_unit[bucket]))
            break_rate = float(np.mean(anchor_unit[bucket] & ~unit_labels[bucket]))
            quintiles.append({"n": int(len(bucket)), "fix": fix, "break": break_rate, "fix_minus_break": fix - break_rate})
    relation = {}
    for label in sorted(set(gold.tolist())):
        selected = gold == label
        relation[str(int(label))] = {"n": int(selected.sum()), "anchor_accuracy": float(np.mean(anchor[selected] == gold[selected])),
                                     "forced_on_accuracy": float(np.mean(forced[selected] == gold[selected]))}
    return {"anchor_vs_forced_on": paired(population, forced, anchor, seed),
            "unit_auroc": _bootstrap_auroc(scorer, unit_labels, unit_groups, seed),
            "post_auroc": _bootstrap_auroc(post_scores, post_labels, post_groups, seed),
            "text_confidence_quintiles": quintiles, "relation_strata": relation,
            "unit_fix_count": int(np.sum(unit_labels & ~anchor_unit)),
            "unit_break_count": int(np.sum(anchor_unit & ~unit_labels)),
            "post_positive_count": int(post_labels.sum()), "post_zero_count": int(np.sum(post_scores == 0))}


def main(ctx, config):
    ctx.tags.update(stage="phase5", study="A4_A5_A9", method="anchor_relative", post_hoc=True, device="cpu")
    sources = json_read(config["sources"])
    records = read_records(sources["manifest"])
    runs_root = Path(config["runs_root"])
    variant = config["variant"]
    trained, trained_row = _find_success(runs_root, ctx.seed, experiment="phase5_training", tags={"variant": variant})
    trained_dev = _rows(trained / "raw/dev.jsonl")
    trained_cal = _rows(trained / "raw/calibration.jsonl")
    trained_test = _rows(trained / "raw/test.jsonl")
    text_root = Path(sources["phase1_root"])
    text_run = _phase1_run(text_root, ctx.seed, "text_only")
    text_cal = _rows(text_run / "raw/calibration.jsonl")
    text_test = _rows(text_run / "raw/test.jsonl")
    text_dev_run, _ = _find_success(sources["phase3_root"], ctx.seed, experiment="phase3_extract", tags={})
    text_dev = _rows(text_dev_run / "raw/independent_dev.jsonl")
    for target in (trained_dev, trained_cal, trained_test):
        for key in target:
            if key not in records:
                raise ValueError(f"Prediction id missing from manifest: {key}")
    anchor_dev = {key: text_dev[key] for key in trained_dev if key in text_dev}
    anchor_cal = {key: text_cal[key] for key in trained_cal if key in text_cal}
    anchor_test = {key: text_test[key] for key in trained_test if key in text_test}
    if len(anchor_dev) != len(trained_dev) or len(anchor_cal) != len(trained_cal) or len(anchor_test) != len(trained_test):
        raise ValueError("Independent text anchor lacks aligned predictions")
    dev = _anchor_rows(trained_dev, anchor_dev)
    cal = _anchor_rows(trained_cal, anchor_cal)
    test = _anchor_rows(trained_test, anchor_test)
    identity = json_read(trained / "data_identity.json")
    populations = calibration_populations(records, identity, ctx.seed)
    maxsim_extract, _ = _find_success(sources["phase4_root"], ctx.seed, experiment="phase4_extract", tags={"method": "maxsim"})
    maxsim_dev_raw = _rows(maxsim_extract / "raw/maxsim_dev.jsonl")
    maxsim_run = _phase1_run(text_root, ctx.seed, "maxsim")
    maxsim_cal_raw = _rows(maxsim_run / "raw/calibration.jsonl")
    maxsim_test_raw = _rows(maxsim_run / "raw/test.jsonl")
    maxsim_dev = _anchor_rows(maxsim_dev_raw, anchor_dev)
    maxsim_cal = _anchor_rows(maxsim_cal_raw, anchor_cal)
    maxsim_test = _anchor_rows(maxsim_test_raw, anchor_test)
    a4 = {"saver": {}, "maxsim": {}, "text_uncertainty_1_minus_max_p": {}}
    for fraction, population in populations.items():
        ids = [row["id"] for row in population]
        fraction_key = str(fraction)
        saver_values = _select_for_fraction(cal, records, ids, dev, test, ctx.seed)
        maxsim_values = _select_for_fraction(maxsim_cal, records, ids, maxsim_dev, maxsim_test, ctx.seed)
        for scorer in a4:
            values = saver_values if scorer == "saver" or scorer == "text_uncertainty_1_minus_max_p" else maxsim_values
            a4[scorer][fraction_key] = values.get(scorer, {"missing": values.get("missing", [])})
    diagnostics = _diagnostics(dev, records, ctx.seed)
    text_uncertainty = _with_scorer(dev, "text_uncertainty_1_minus_max_p")
    saver = _with_scorer(dev, "saver")
    auroc_compare = {"saver": diagnostics["post_auroc"],
                     "text_uncertainty_1_minus_max_p": _bootstrap_auroc(
                         np.asarray([row["gate"][0] for row in text_uncertainty.values()]),
                         np.asarray([(row["vision_prediction"][0] == row["gold"][0]) and
                                     (row["text_prediction"][0] != row["gold"][0])
                                     for row in text_uncertainty.values()]),
                         _post_groups(text_uncertainty, records), ctx.seed)}
    a9 = {}
    for scorer, fractions in a4.items():
        a9[scorer] = {}
        for fraction, caps in fractions.items():
            if "missing" in caps:
                continue
            a9[scorer][fraction] = {}
            for cap, choice in caps.items():
                target_rate = choice["test"]["coverage"]
                random_rows = []
                for random_seed in (2401, 2402, 2403, 2404, 2405):
                    population = _population(_with_scorer(test, scorer), records)
                    active = population.eligible.copy()
                    active[:] = False
                    eligible = np.flatnonzero(population.eligible)
                    count = min(len(eligible), int(round(target_rate * len(population.rows))))
                    chosen = np.random.default_rng(random_seed).permutation(eligible)[:count]
                    active[chosen] = True
                    random_rows.append(population.metrics(np.where(active, population.vision, population.text), active))
                a9[scorer][fraction][cap] = {"target_rate": target_rate, "seeds": random_rows,
                                              "mean_f1": float(np.mean([row["f1"] for row in random_rows])),
                                              "mean_harm": float(np.mean([row["harm"] for row in random_rows]))}
    ctx.save_json("phase5_replay.json", {"variant": variant, "seed": ctx.seed, "trained_run_id": trained_row["run_id"],
                  "a4": a4, "a5": diagnostics, "auroc_compare": auroc_compare, "a9": a9,
                  "test_rule": "descriptive only; no test-based selection"})
    ctx.set_metrics({"variant": variant, "a4": a4, "a5": diagnostics, "auroc_compare": auroc_compare, "a9": a9,
                     "training_updates": 0, "gpu_hours_actual": 0., "new_test_forward_calls": 0})
