"""CPU replay for P24 dual certificates, matched baselines, and fallback rates."""

from pathlib import Path

import numpy as np

from .phase4_replay import (ALPHA, CAPS, DELTA, FAMILY_SIZE, Phase4Population, calibration_ids, choose_dual,
                            choose_reference, evaluate_choice, find_run, final_epoch, json_read,
                            make_population, method_paths, policies, random_mask, rows_for,
                            calibrate_dual)
from .protocol import digest, read_records


def cap_key(cap):
    return f"c{int(round(cap * 100)):02d}"


def compact_metric(metric):
    keep = ("f1", "harm", "cost", "coverage", "detail", "latency_ms", "p90_ms", "n_posts", "n_units",
            "harmful_posts", "positive_harm_rate", "gain")
    return {key: metric.get(key) for key in keep}


def compact_choice(choice):
    return {"id": choice["id"], "threshold": choice["threshold"], "eta": choice["eta"],
            "harm_ucb": choice["harm_ucb"], "cost_ucb": choice["cost_ucb"],
            "calibration_harm": choice["calibration_harm"], "calibration_cost": choice["calibration_cost"],
            "fallback": choice["fallback"], "feasible_nonempty": choice["feasible_nonempty"],
            "f1_regret": choice["f1_regret"], "f1_delta_vs_text": choice["f1_delta_vs_text"],
            "test_f1_ge_text": choice.get("test_f1_ge_text"),
            "dev": compact_metric(choice["dev"]), "test": compact_metric(choice["test"])}


def policy_row(policy, development, test):
    return {"id": policy["id"], "threshold": policy["threshold"], "harm_ucb": policy["harm_ucb"],
            "cost_ucb": policy["cost_ucb"], "harm_feasible": policy["harm_feasible"],
            "cost_feasible": policy["cost_feasible"], "dual_feasible": policy["dual_feasible"],
            "calibration": compact_metric(policy), "dev": compact_metric(development.evaluate(policy)),
            "test": compact_metric(test.evaluate(policy))}


def load_manifest(config, saver_run):
    source = json_read(saver_run / "config.json")
    manifest = Path(config["source_repo"]) / source["manifest"]
    return read_records(manifest), manifest


def calibration_rows(info, method):
    rows = rows_for(info, "calibration")
    if method == "saver":
        extension = info.get("extension")
        if extension is not None and extension.exists():
            rows += read_records(extension)
    return {row["id"]: row for row in rows}


def population_for(records, row_index):
    missing = [row["id"] for row in records if row["id"] not in row_index]
    if missing:
        return None, missing
    return [row_index[row["id"]] for row in records], []


def main(ctx, config):
    ctx.tags.update(stage="phase4", study="D1_D2_D3", method="dual_certificate", device="cpu")
    sources = method_paths(config, ctx.seed)
    saver_run = sources["saver"]["run"]
    manifest, manifest_path = load_manifest(config, saver_run)
    identity = json_read(saver_run / "data_identity.json")
    populations = calibration_ids(manifest, identity, ctx.seed)
    timings = None
    try:
        phase2 = Path(config["phase2_root"])
        timing_run, _ = find_run(phase2, ctx.seed, experiment="batch_serving")
        timing_rows = [row for row in read_records(timing_run / "raw/batch_timings.jsonl")
                       if row.get("status") == "OK" and row.get("batch_size") == 1 and not row.get("bucketed")]
        off = [row for row in timing_rows if row.get("method") == "saver_k1"]
        on = [row for row in timing_rows if row.get("method") == "always_top1"]
        timings = {"off_ms": float(np.mean([row["latency_ms"] for row in off])),
                   "on_ms": float(np.mean([row["latency_ms"] for row in on]))}
    except Exception:
        timings = None

    for method, info in sources.items():
        if method == "text_only":
            continue
        info["extension"] = Path(config["extraction_run"]) / "raw/calibration_extension.jsonl" if method == "saver" else None
        info["calibration_index"] = calibration_rows(info, method)
        info["dev_rows"] = rows_for(info, "dev")
        info["test_rows"] = rows_for(info, "test")

    text_info = sources["text_only"]
    independent_test = {row["id"]: row for row in rows_for(text_info, "test")}
    metrics = {"d1": {}, "d2": {}, "d3": {}, "random_gate": {}, "missing": {},
               "source": {}, "training_updates": 0, "gpu_hours_actual": 0., "new_test_forward_calls": 0}
    frontier = []
    missing = []

    for method in ("saver", "maxsim", "linear", "utility_mlp_k1"):
        info = sources[method]
        metrics["source"][method] = {"run_id": info["row"]["run_id"],
                                      "run_dir": str(info["run"]),
                                      "dev_sha256": digest(info["dev"]),
                                      "calibration_sha256": digest(info["run"] / "raw/calibration.jsonl"),
                                      "test_sha256": digest(info["run"] / "raw/test.jsonl")}
        dev = make_population(info["dev_rows"], manifest, timings)
        test = make_population(info["test_rows"], manifest, timings)
        independent = np.array([independent_test[key]["prediction"][0] for key in test.ids])
        if any(independent_test[key]["gold"][0] != label for key, label in zip(test.ids, test.gold)):
            raise ValueError("Independent text and baseline test gold labels differ")
        baseline_text_f1 = test.metrics(independent)["f1"]
        method_d1 = {}
        method_d2 = {}
        for cap in CAPS:
            curve = calibrate_dual(make_population(population_for(populations[0.10], info["calibration_index"])[0], manifest, timings), policies(), cap) \
                if population_for(populations[0.10], info["calibration_index"])[0] is not None else None
            if curve is None:
                missing.append({"method": method, "seed": ctx.seed, "fraction": 0.10, "cap": cap,
                                "reason": "calibration logits missing for requested posts"})
                continue
            selected, fallback, feasible = choose_dual(curve, dev)
            choice = evaluate_choice(test, dev, test, selected, fallback, feasible)
            choice["f1_delta_vs_text"] = float(choice["test"]["f1"] - baseline_text_f1)
            choice["test_f1_ge_text"] = int(choice["test"]["f1"] + 1e-12 >= baseline_text_f1)
            choice["any_feasible_test_f1_ge_text"] = bool(any(test.evaluate(row)["f1"] + 1e-12 >= baseline_text_f1 for row in feasible))
            method_d2[cap_key(cap)] = compact_choice(choice)
            if method == "saver":
                method_d1[cap_key(cap)] = compact_choice(choice)
                for selector in ("harm_only", "max_dev_f1", "cost_penalty"):
                    reference, reference_fallback = choose_reference(curve, dev, selector)
                    reference_choice = evaluate_choice(test, dev, test, reference, reference_fallback,
                                                       [row for row in curve if row["harm_feasible"] and row["coverage"] > 0])
                    reference_choice["f1_delta_vs_text"] = float(reference_choice["test"]["f1"] - baseline_text_f1)
                    reference_choice["test_f1_ge_text"] = int(reference_choice["test"]["f1"] + 1e-12 >= baseline_text_f1)
                    method_d1[cap_key(cap)][selector] = compact_choice(reference_choice)
            for policy in curve:
                frontier.append({"method": method, "seed": ctx.seed, "cap": cap_key(cap),
                                 **policy_row(policy, dev, test)})
        metrics["d2"][method] = method_d2
        if method == "saver":
            metrics["d1"] = method_d1

        d3_method = {}
        for fraction in (0.02, 0.05, 0.10, 0.20):
            fraction_key = f"c{int(round(fraction * 100)):02d}"
            d3_method[fraction_key] = {}
            selected_rows, missing_ids = population_for(populations[fraction], info["calibration_index"])
            if missing_ids:
                for cap in CAPS:
                    d3_method[fraction_key][cap_key(cap)] = {"status": "missing", "missing_ids": len(missing_ids),
                                                               "nonempty_feasible": None, "fallback": None}
                missing.append({"method": method, "seed": ctx.seed, "fraction": fraction,
                                "reason": "existing calibration logits do not cover this population",
                                "missing_ids": len(missing_ids)})
                continue
            calibration = make_population(selected_rows, manifest, timings)
            for cap in CAPS:
                curve = calibrate_dual(calibration, policies(), cap)
                feasible_count = sum(row["dual_feasible"] and row["coverage"] > 0 for row in curve)
                d3_method[fraction_key][cap_key(cap)] = {
                    "status": "ok", "calibration_posts": len(calibration.post_ids),
                    "feasible_policies": int(feasible_count), "feasible_rate": float(feasible_count / FAMILY_SIZE),
                    "nonempty_feasible": int(feasible_count > 0), "fallback": int(feasible_count == 0),
                    "cost_cap": cap,
                }
        metrics["d3"][method] = d3_method

    saver_info = sources["saver"]
    saver_test = make_population(saver_info["test_rows"], manifest, timings)
    for cap in CAPS:
        saver_choice = metrics["d1"][cap_key(cap)]["id"]
        saver_policy = next(row for row in policies() if row["id"] == saver_choice)
        selected_active = saver_test.active(saver_policy)
        target_rate = float(selected_active.mean())
        samples = []
        for random_seed in (2401, 2402, 2403, 2404, 2405):
            active = random_mask(saver_test, target_rate, random_seed)
            samples.append(compact_metric(saver_test.metrics(saver_test.text * 0 + np.where(active, saver_test.vision, saver_test.text), active)))
        metrics["random_gate"][cap_key(cap)] = {"target_activation_rate": target_rate,
            "random_seeds": [2401, 2402, 2403, 2404, 2405], "test": samples,
            "f1_mean": float(np.mean([item["f1"] for item in samples])),
            "f1_min": float(min(item["f1"] for item in samples)),
            "f1_max": float(max(item["f1"] for item in samples)),
            "harm_mean": float(np.mean([item["harm"] for item in samples])),
            "cost_mean": float(np.mean([item["cost"] for item in samples]))}

    metrics["missing"] = missing
    metrics["calibration_rule"] = {"family_size": FAMILY_SIZE, "alpha": ALPHA, "delta": DELTA,
                                    "cost_ucb": "Hoeffding one-sided, delta/(2*family_size)",
                                    "harm_ucb": "Clopper-Pearson one-sided, delta/(2*family_size)"}
    metrics["independent_text_test_f1"] = float(test.metrics(independent)["f1"])
    metrics["frontier_rows"] = len(frontier)
    ctx.save_json("dual_frontier.json", frontier)
    ctx.save_json("method_sources.json", metrics["source"])
    ctx.save_json("d3_frequency.json", metrics["d3"])
    ctx.save_json("random_gate.json", metrics["random_gate"])
    ctx.save_json("missing_logits.json", missing)
    ctx.save_json("selection_audit.json", {"test_forward_calls": 0, "new_test_labels_opened": False,
                  "numeric_tables_generator": "python -m repro.make_tables", "policy_family_size": FAMILY_SIZE,
                  "caps": list(CAPS)})
    ctx.set_metrics(metrics)
