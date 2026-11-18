"""Replay helpers for the preregistered P24 phase4 dual certificates."""

import math
import random
from pathlib import Path

import numpy as np

from .calibration import cp_upper
from .phase3_replay import Population, calibration_populations, fingerprint, json_read
from .protocol import read_records


CAPS = (0.25, 0.35, 0.50)
FAMILY_SIZE = 101
ALPHA = 0.10
DELTA = 0.05


def find_run(root, seed, *, experiment, tag_key=None, tag_value=None, config_key=None, config_value=None, extra_tags=None):
    root = Path(root)
    matches = []
    for row in read_records(root / "ledger.jsonl"):
        tags = row.get("tags", {})
        if row.get("paper") != "P24" or row.get("experiment") != experiment:
            continue
        if row.get("seed") != seed or row.get("status") != "SUCCESS":
            continue
        if tag_key is not None and tags.get(tag_key) != tag_value:
            continue
        if extra_tags and any(tags.get(key) != value for key, value in extra_tags.items()):
            continue
        if config_key is not None:
            config = json_read(root / row["run_dir"] / "config.json")
            if config.get(config_key) != config_value:
                continue
        matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one {experiment} run for seed={seed}, filters={tag_key}:{tag_value},{config_key}:{config_value}; found {len(matches)}")
    return root / matches[0]["run_dir"], matches[0]


def method_paths(config, seed):
    phase1 = Path(config["phase1_root"])
    phase2 = Path(config["phase2_root"])
    phase3 = Path(config["phase3_root"])
    result = {}
    saver, saver_row = find_run(phase1, seed, experiment="experiment", tag_key="method", tag_value="saver")
    text, text_row = find_run(phase1, seed, experiment="experiment", tag_key="method", tag_value="text_only")
    maxsim, maxsim_row = find_run(phase1, seed, experiment="experiment", tag_key="method", tag_value="maxsim")
    linear, linear_row = find_run(phase2, seed, experiment="phase2_training", tag_key="router", tag_value="linear", extra_tags={"stage": "formal"})
    utility, utility_row = find_run(phase2, seed, experiment="phase2_training", tag_key="router", tag_value="utility_mlp", extra_tags={"stage": "formal"})
    saver_audit, saver_audit_row = find_run(phase2, seed, experiment="phase2_audit", extra_tags={"parent_run_id": saver_row["run_id"]})
    result["saver"] = {"run": saver, "row": saver_row, "label": "SAVER", "test_path": saver_audit / "raw/frozen_test.jsonl", "test_audit_row": saver_audit_row}
    result["maxsim"] = {"run": maxsim, "row": maxsim_row, "label": "MaxSim"}
    result["linear"] = {"run": linear, "row": linear_row, "label": "Linear"}
    result["utility_mlp_k1"] = {"run": utility, "row": utility_row, "label": "UtilityMLP-K1"}
    result["text_only"] = {"run": text, "row": text_row, "label": "ModernBERT-only"}
    extraction = Path(config["extraction_run"])
    result["saver"]["dev"] = extraction / "raw/saver_dev.jsonl"
    maxsim_extract, maxsim_extract_row = find_run(Path(config["maxsim_extraction_run"]), seed, experiment="phase4_extract", tag_key="method", tag_value="maxsim")
    result["maxsim"]["dev"] = maxsim_extract / "raw/maxsim_dev.jsonl"
    result["maxsim"]["extraction_row"] = maxsim_extract_row
    result["linear"]["dev"] = linear / "raw/dev_soft.jsonl"
    result["utility_mlp_k1"]["dev"] = utility / "raw/dev_soft.jsonl"
    result["text_only"]["dev"] = extraction / "raw/independent_dev.jsonl"
    return result


def final_epoch(path):
    rows = read_records(path)
    if not rows:
        raise ValueError(f"Empty prediction log: {path}")
    epochs = [row.get("epoch") for row in rows]
    if any(epoch is not None for epoch in epochs):
        last = max(epoch for epoch in epochs if epoch is not None)
        rows = [row for row in rows if row.get("epoch") == last]
    normalized = []
    for row in rows:
        copied = dict(row)
        copied["selected"] = [[0] for _ in copied.get("gold", [0])]
        normalized.append(copied)
    rows = normalized
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate prediction IDs after final-epoch selection: {path}")
    return list(by_id.values())


def rows_for(info, split):
    if split == "dev":
        path = info["dev"]
    elif split == "calibration":
        path = info["run"] / "raw/calibration.jsonl"
    elif split == "test":
        path = info.get("test_path", info["run"] / "raw/test.jsonl")
    else:
        raise ValueError(split)
    return final_epoch(path)


def base_timings(config, seed):
    phase2 = Path(config["phase2_root"])
    row_dir, row = find_run(phase2, seed, experiment="batch_serving")
    rows = [row for row in read_records(row_dir / "raw/batch_timings.jsonl")
            if row.get("status") == "OK" and row.get("batch_size") == 1 and not row.get("bucketed")]
    groups = {}
    for timing in rows:
        groups.setdefault(timing["method"], []).append(timing)
    off, on = groups["saver_k1"], groups["always_top1"]
    return {"off_ms": float(np.mean([row["latency_ms"] for row in off])),
            "on_ms": float(np.mean([row["latency_ms"] for row in on]))}


class Phase4Population(Population):
    def metrics(self, prediction, active=None):
        result = super().metrics(prediction, active)
        active = np.zeros(len(prediction), dtype=bool) if active is None else active
        values = np.zeros(len(self.post_ids), dtype=float)
        seen = [set() for _ in self.post_ids]
        for index in np.flatnonzero(active):
            selected = self.rows[index].get("selected", [[0]])[0]
            selected = [value for value in selected if value >= 0]
            for attachment in selected:
                if attachment < len(self.images[index]):
                    seen[self.groups[index]].add(self.images[index][attachment])
        for post_index, images in enumerate(self.images_for_posts()):
            values[post_index] = len(seen[post_index]) / max(1, len(images))
        result["cost_values"] = values.tolist()
        result["cost"] = float(values.mean())
        return result

    def images_for_posts(self):
        result = [[] for _ in self.post_ids]
        for group, images in zip(self.groups, self.images):
            result[group] = images
        return result


def make_population(rows, manifest, timings):
    return Phase4Population(rows, manifest, timings)


def cost_radius(n):
    return math.sqrt(math.log(2 * FAMILY_SIZE / DELTA) / (2 * n))


def calibrate_dual(population, policies, cap):
    if not population.post_ids:
        raise ValueError("Calibration population has no posts")
    result = []
    confidence = 1 - DELTA / (2 * FAMILY_SIZE)
    radius = cost_radius(len(population.post_ids))
    for policy in policies:
        metric = dict(population.evaluate(policy))
        if policy["threshold"] is None:
            harm_ucb = 0.0
            cost_ucb = 0.0
        else:
            harm_ucb = cp_upper(metric["harmful_posts"], metric["n_posts"], confidence)
            cost_ucb = min(1.0, metric["cost"] + radius)
        result.append({**policy, **metric, "harm_ucb": harm_ucb, "cost_ucb": cost_ucb,
                       "dual_feasible": bool(harm_ucb <= ALPHA and cost_ucb <= cap),
                       "harm_feasible": bool(harm_ucb <= ALPHA),
                       "cost_feasible": bool(cost_ucb <= cap)})
    return result


def policies():
    return [{"id": f"t{index:03d}", "eta": 0.0, "threshold": index / 100} for index in range(101)] + [
        {"id": "off", "eta": 0.0, "threshold": None}]


def choose_dual(curve, development):
    feasible = [row for row in curve if row["dual_feasible"] and row["coverage"] > 0]
    if not feasible:
        return next(row for row in curve if row["id"] == "off"), True, feasible
    ranked = []
    for row in feasible:
        metric = development.evaluate(row)
        threshold = 2.0 if row["threshold"] is None else row["threshold"]
        ranked.append((metric["f1"], -metric["cost"], threshold, row["id"], row))
    return max(ranked, key=lambda item: item[:4])[4], False, feasible


def choose_reference(curve, development, selector):
    feasible = [row for row in curve if row["harm_feasible"]]
    nonempty = [row for row in feasible if row["coverage"] > 0]
    if not nonempty:
        return next(row for row in curve if row["id"] == "off"), True
    scored = []
    for row in feasible:
        metric = development.evaluate(row)
        threshold = 2.0 if row["threshold"] is None else row["threshold"]
        if selector == "harm_only":
            key = (row["coverage"], -threshold, row["id"])
        elif selector == "max_dev_f1":
            key = (metric["f1"], -metric["cost"], -row["harm_ucb"], threshold, row["id"])
        elif selector == "cost_penalty":
            key = (metric["f1"] - 0.10 * metric["cost"], -metric["cost"], -row["harm_ucb"], threshold, row["id"])
        else:
            raise ValueError(selector)
        scored.append((key, row))
    return max(scored, key=lambda item: item[0])[1], False


def enrich(selected, test, candidates):
    return {**selected, "test": test,
            "f1_regret": float(max(row["test"]["f1"] for row in candidates) - test["f1"]),
            "selected_test_cost": test["cost"]}


def evaluate_choice(population, development, test, choice, fallback, candidates):
    policy = choice
    dev = development.evaluate(policy)
    test_metric = test.evaluate(policy)
    candidate_f1 = [row["test"]["f1"] if "test" in row else test.evaluate(row)["f1"] for row in candidates]
    f1_regret = 0.0 if not candidate_f1 else max(candidate_f1) - test_metric["f1"]
    return {"id": policy["id"], "threshold": policy["threshold"], "eta": policy["eta"],
            "harm_ucb": policy["harm_ucb"], "cost_ucb": policy["cost_ucb"],
            "calibration_harm": policy["harm"], "calibration_cost": policy["cost"],
            "fallback": int(fallback), "dev": dev, "test": test_metric,
            "f1_regret": float(f1_regret),
            "f1_delta_vs_text": None,
            "feasible_nonempty": int(bool(candidates))}


def random_mask(population, target_rate, seed):
    eligible = np.flatnonzero(population.eligible)
    count = int(round(target_rate * len(population.rows)))
    count = min(len(eligible), max(0, count))
    generator = np.random.default_rng(seed)
    chosen = generator.permutation(eligible)[:count]
    active = np.zeros(len(population.rows), dtype=bool)
    active[chosen] = True
    return active


def calibration_ids(manifest, identity, seed):
    return calibration_populations(manifest, identity, seed)
