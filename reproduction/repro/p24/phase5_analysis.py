"""Complete anchor-relative replay; only dev and calibration determine decisions."""

from pathlib import Path

import numpy as np

from .phase3_replay import auroc, calibration_populations, json_read, paired
from .phase4_replay import CAPS, Phase4Population, calibrate_dual, choose_dual, final_epoch, find_run, policies, random_mask
from .protocol import digest, read_records


SCORERS = ("saver", "maxsim", "uncertainty")
SEEDS_RANDOM = (2401, 2402, 2403, 2404, 2405)


def index(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated prediction IDs")
    return result


def parent(sources, seed, method):
    return find_run(sources["phase1_root"], seed, experiment="experiment", tag_key="method", tag_value=method,
                    extra_tags={"stage": "formal"})[0]


def extract_run(sources, seed):
    return find_run(sources["phase3_root"], seed, experiment="phase3_extract")[0]


def align(branch, anchor, *, scorer="saver", score_rows=None):
    branch = index(branch)
    anchor = index(anchor)
    score_rows = branch if score_rows is None else index(score_rows)
    result = []
    for key in sorted(branch):
        row, text = branch[key], anchor[key]
        if row["gold"] != text["gold"]:
            raise ValueError(f"Gold mismatch: {key}")
        score = row["gate"][0]
        if scorer == "maxsim":
            score = score_rows[key]["gate"][0]
        elif scorer == "uncertainty":
            probabilities = text.get("text_probabilities", text.get("probabilities"))
            if probabilities is None:
                raise ValueError(f"Missing independent text probabilities: {key}")
            score = 1 - max(probabilities[0])
        result.append({**row, "text_prediction": text["prediction"], "gate": [score],
                       "selected": [[0] if eligible else [] for eligible in row["eligible"]]})
    return result


def population(rows, manifest):
    return Phase4Population(sorted(rows, key=lambda row: row["id"]), manifest, None)


def subset(rows, ids):
    rows = index(rows)
    missing = sorted(set(ids) - set(rows))
    return None if missing else [rows[key] for key in sorted(ids)], missing


def bootstrap_auc(population_, seed):
    gains_unit = (population_.vision == population_.gold).astype(int) - (population_.text == population_.gold).astype(int)
    gains_post = np.bincount(population_.groups, weights=gains_unit) / population_.counts
    score_post = np.bincount(population_.groups, weights=population_.gate) / population_.counts
    labels_unit, labels_post = gains_unit > 0, gains_post > 0
    post_members = [np.flatnonzero(population_.groups == group) for group in range(len(population_.post_ids))]
    generator = np.random.default_rng(20261114 + seed)
    samples = {"unit": [], "post": []}
    for _ in range(2000):
        posts = generator.integers(0, len(post_members), len(post_members))
        units = np.concatenate([post_members[group] for group in posts])
        for name, scores, labels, selected in (("unit", population_.gate, labels_unit, units),
                                               ("post", score_post, labels_post, posts)):
            value = auroc(scores[selected], labels[selected])
            if value is not None:
                samples[name].append(value)
    result = {}
    for name, scores, labels in (("unit", population_.gate, labels_unit), ("post", score_post, labels_post)):
        values = samples[name]
        result[name] = {"auroc": auroc(scores, labels), "ci_low": float(np.quantile(values, .025)) if values else None,
                        "ci_high": float(np.quantile(values, .975)) if values else None, "valid_bootstraps": len(values),
                        "brier": float(np.mean((scores - labels) ** 2)), "n": len(labels), "positive": int(labels.sum())}
    result["unit"].update(negative=int(np.sum(gains_unit < 0)), zero=int(np.sum(gains_unit == 0)))
    result["post"].update(negative=int(np.sum(gains_post < 0)), zero=int(np.sum(gains_post == 0)))
    return result


def random_control(population_, policy):
    rate = float(population_.active(policy).mean())
    values = []
    for seed in SEEDS_RANDOM:
        active = random_mask(population_, rate, seed)
        metric = population_.metrics(np.where(active, population_.vision, population_.text), active)
        values.append({"random_seed": seed, **compact(metric)})
    summary = {key: {"mean": float(np.mean([row[key] for row in values])),
                     "min": min(row[key] for row in values), "max": max(row[key] for row in values)}
               for key in ("f1", "harm", "coverage", "detail", "cost")}
    return {"target_rate": rate, "values": values, "draws": {f"s{row['random_seed']}": row for row in values},
            "summary": summary,
            "f1_mean": float(np.mean([row["f1"] for row in values])), "f1_min": min(row["f1"] for row in values),
            "f1_max": max(row["f1"] for row in values), "harm_mean": float(np.mean([row["harm"] for row in values])),
            "coverage_mean": float(np.mean([row["coverage"] for row in values])),
            "cost_mean": float(np.mean([row["cost"] for row in values]))}


def compact(metric):
    keys = ("f1", "harm", "positive_harm_rate", "harmful_posts", "gain", "detail", "coverage", "cost", "n_posts", "n_units")
    return {key: metric[key] for key in keys}


def sensitivity(branches, anchors, manifest, populations, seed, *, maxsim_scores=None):
    result, locks, active_random = {}, {}, {}
    for scorer in SCORERS:
        if scorer not in branches:
            result[scorer] = {"status": "not_applicable", "reason": "No frozen branch or scorer predictions"}
            continue
        branch = branches[scorer]
        dev_rows = align(branch["dev"], anchors["dev"], scorer=scorer,
                         score_rows=None if maxsim_scores is None else maxsim_scores.get("dev"))
        dev = population(dev_rows, manifest)
        result[scorer] = {}
        locks[scorer] = {}
        active_random[scorer] = {}
        for fraction, records in populations.items():
            fraction_key = f"f{round(fraction * 100):02d}"
            ids = [row["id"] for row in records]
            raw, missing_branch = subset(branch["calibration"], ids)
            anchor, missing_anchor = subset(anchors["calibration"], ids)
            score, missing_score = (None, []) if maxsim_scores is None else subset(maxsim_scores["calibration"], ids)
            if missing_branch or missing_anchor or (scorer == "maxsim" and missing_score):
                result[scorer][fraction_key] = {"status": "not_applicable", "missing_branch": len(missing_branch),
                                                 "missing_anchor": len(missing_anchor), "missing_score": len(missing_score)}
                continue
            cal = population(align(raw, anchor, scorer=scorer, score_rows=score), manifest)
            result[scorer][fraction_key] = {}
            locks[scorer][fraction_key] = {}
            active_random[scorer][fraction_key] = {}
            for cap in CAPS:
                cap_key = f"c{round(cap * 100):02d}"
                curve = calibrate_dual(cal, policies(), cap)
                choice, fallback, feasible = choose_dual(curve, dev)
                dev_metric = dev.evaluate(choice)
                random_cal = None if fallback else random_control(cal, choice)
                random_dev = None if fallback else random_control(dev, choice)
                row = {"threshold": choice["threshold"], "harm_ucb": choice["harm_ucb"], "cost_ucb": choice["cost_ucb"],
                       "calibration": compact(choice), "dev": compact(dev_metric), "fallback": int(fallback),
                       "feasible_policies": len(feasible), "feasible_nonempty": int(bool(feasible)),
                       "random_calibration": random_cal, "random_dev": random_dev,
                       "random_status": "not_applicable_structural_all_off" if fallback else "five_fixed_draws",
                       "better_random_harm": int(random_cal is not None and choice["harm"] < random_cal["harm_mean"] - 1e-12),
                       "better_random_dev_f1": int(random_dev is not None and dev_metric["f1"] > random_dev["f1_mean"] + 1e-12),
                       "dev_paired_anchor": paired(dev, np.where(dev.active(choice), dev.vision, dev.text), dev.text, seed)}
                result[scorer][fraction_key][cap_key] = row
                locks[scorer][fraction_key][cap_key] = {key: choice[key] for key in ("id", "threshold", "eta")}
            endpoints = {"off": {"id": "off", "threshold": None, "eta": 0.}, "on": {"id": "on", "threshold": 0., "eta": 0.}}
            result[scorer][fraction_key]["endpoints"] = {
                name: {"calibration": compact(cal.evaluate(policy)), "dev": compact(dev.evaluate(policy))}
                for name, policy in endpoints.items()}
    return result, locks


def score_test(results, locks, branches, anchors, manifest):
    for scorer, fractions in locks.items():
        test = population(align(branches[scorer]["test"], anchors["test"], scorer=scorer), manifest)
        anchor_f1 = test.metrics(test.text)["f1"]
        for fraction_key, caps in fractions.items():
            for cap_key, policy in caps.items():
                metric = compact(test.evaluate(policy))
                row = results[scorer][fraction_key][cap_key]
                row.update(test=metric, test_f1_delta_anchor=metric["f1"] - anchor_f1,
                           random_test=random_control(test, policy) if row["dev"]["detail"] > 0 else None)
            for name, threshold in (("off", None), ("on", 0.)):
                results[scorer][fraction_key]["endpoints"][name]["test"] = compact(test.evaluate({"threshold": threshold, "eta": 0.}))


def study_sources(config, seed, *, sources=None):
    if sources is None:
        sources = json_read(config["sources"])
    study = config["study"]
    if study in ("A1", "A2", "A7"):
        trained, row = find_run(config["phase5_root"], seed, experiment="phase5_training", tag_key="study", tag_value=study)
        branch = {split: final_epoch(trained / f"raw/{split}.jsonl") for split in ("dev", "calibration")}
        identity = json_read(trained / "data_identity.json")
        text = parent(sources, seed, "text_only")
        if study == "A7":
            text, _ = find_run(config["phase5_root"], seed, experiment="phase5_training", tag_key="study", tag_value="A6",
                               extra_tags={"method": "text_only"})
        scope = "full" if study != "A7" else "uncapped_pilot" if config.get("pilot") else "uncapped"
        anchor_extract, _ = find_run(config["phase5_root"], seed, experiment="phase5_anchor", tag_key="scope", tag_value=scope)
        anchors = {"dev": final_epoch(anchor_extract / "raw/dev.jsonl"),
                   "calibration": final_epoch(text / "raw/calibration.jsonl")}
        if config.get("include_extension"):
            extension, _ = find_run(config["phase5_root"], seed, experiment="phase5_extension", tag_key="study", tag_value=study)
            branch["calibration"] += final_epoch(extension / "raw/calibration_extension.jsonl")
            anchors["calibration"] += final_epoch(anchor_extract / "raw/calibration_extension.jsonl")
        anchors["dev"], missing = subset(anchors["dev"], identity["split_ids"]["dev"])
        if missing:
            raise ValueError(f"Missing anchor dev rows: {missing[:3]}")
        branches = {"saver": branch, "uncertainty": branch}
        maxsim_path = parent(sources, seed, "maxsim")
        if study == "A7" and config.get("pilot"):
            return sources, trained, row, identity, anchors, branches, text, None
        if study == "A7":
            maxsim_path, _ = find_run(config["phase5_root"], seed, experiment="phase5_training", tag_key="study", tag_value="A7_maxsim")
        maxsim_extract = anchor_extract
        score_dev, missing = subset(final_epoch(maxsim_extract / "raw/maxsim_dev.jsonl"), identity["split_ids"]["dev"])
        if not missing:
            branches["maxsim"] = {"dev": score_dev, "calibration": final_epoch(maxsim_path / "raw/calibration.jsonl")}
            if config.get("include_extension"):
                branches["maxsim"]["calibration"] += final_epoch(anchor_extract / "raw/maxsim_extension.jsonl")
        branches["maxsim"]["test_path"] = anchor_extract / "raw/maxsim_test.jsonl"
        return sources, trained, row, identity, anchors, branches, text, maxsim_path
    if study != "legacy":
        raise ValueError(study)
    trained = parent(sources, seed, "saver")
    extraction = extract_run(sources, seed)
    branch = {"dev": final_epoch(extraction / "raw/saver_dev.jsonl"),
              "calibration": final_epoch(trained / "raw/calibration.jsonl") + final_epoch(extraction / "raw/calibration_extension.jsonl")}
    text = parent(sources, seed, "text_only")
    anchor_extract, _ = find_run(config["phase5_root"], seed, experiment="phase5_anchor", tag_key="scope", tag_value="full")
    anchors = {"dev": final_epoch(anchor_extract / "raw/dev.jsonl"),
               "calibration": final_epoch(text / "raw/calibration.jsonl") + final_epoch(anchor_extract / "raw/calibration_extension.jsonl")}
    maxsim_path = parent(sources, seed, "maxsim")
    maxsim_extract, _ = find_run(config["phase5_root"], seed, experiment="phase5_anchor", tag_key="scope", tag_value="full")
    branches = {"saver": branch, "uncertainty": branch,
                "maxsim": {"dev": final_epoch(maxsim_extract / "raw/maxsim_dev.jsonl"),
                           "calibration": final_epoch(maxsim_path / "raw/calibration.jsonl") + final_epoch(maxsim_extract / "raw/maxsim_extension.jsonl"),
                           "test_path": maxsim_extract / "raw/maxsim_test.jsonl"}}
    identity = json_read(trained / "data_identity.json")
    return sources, trained, {}, identity, anchors, branches, text, maxsim_path


def main(ctx, config):
    if __import__("torch").cuda.is_available():
        raise RuntimeError("CPU analysis requires CUDA_VISIBLE_DEVICES empty")
    study = config["study"]
    ctx.tags.update(stage="phase5", study=study, analysis="anchor_certificate", analysis_revision=config.get("analysis_revision", "v3"), post_hoc=True, device="cpu")
    sources, trained, trained_row, identity, anchors, branches, text, maxsim_path = study_sources(config, ctx.seed)
    manifest = read_records(sources["manifest"])
    fractions = (.02, .05, .1) if study == "A7" else (.02, .05, .1, .2)
    populations = calibration_populations(manifest, identity, ctx.seed, fractions=fractions)
    ctx.save_json("source_identity.json", {"trained_run": str(trained), "anchor_run": str(text),
                  "manifest_sha256": digest(sources["manifest"]), "checkpoint_sha256": digest(trained / "best.pt"),
                  "anchor_sha256": digest(text / "best.pt")})
    results, locks = sensitivity(branches, anchors, manifest, populations, ctx.seed)
    cal_raw, _ = subset(branches["saver"]["calibration"], identity["split_ids"]["calibration"])
    cal_anchor, missing = subset(anchors["calibration"], identity["split_ids"]["calibration"])
    if missing:
        raise ValueError("Calibration anchor incomplete")
    saver_cal = population(align(cal_raw, cal_anchor), manifest)
    uncertainty_cal = population(align(cal_raw, cal_anchor, scorer="uncertainty"), manifest)
    auc = {"saver": bootstrap_auc(saver_cal, ctx.seed), "uncertainty": bootstrap_auc(uncertainty_cal, ctx.seed)}
    if "maxsim" in branches:
        raw, _ = subset(branches["maxsim"]["calibration"], identity["split_ids"]["calibration"])
        auc["maxsim"] = bootstrap_auc(population(align(raw, cal_anchor, scorer="maxsim"), manifest), ctx.seed)
    dev_own = population(branches["saver"]["dev"], manifest)
    dev_anchor = population(align(branches["saver"]["dev"], anchors["dev"]), manifest)
    if dev_own.ids != dev_anchor.ids:
        raise ValueError("Paired dev prediction IDs differ")
    fallback = paired(dev_own, dev_own.text, dev_anchor.text, ctx.seed)
    forced = paired(dev_anchor, dev_anchor.vision, dev_anchor.text, ctx.seed)
    nontrivial_thresholds = sum(0 < dev_own.evaluate(policy)["detail"] < 1 for policy in policies() if policy["threshold"] is not None)
    frontier = {policy["id"]: {"threshold": policy["threshold"], "own_fallback": compact(dev_own.evaluate(policy)),
                              "capable_fallback": compact(dev_anchor.evaluate(policy))} for policy in policies()}
    detail_values = [row["own_fallback"]["detail"] for row in frontier.values() if row["threshold"] is not None]
    histogram = {"zero": sum(value == 0 for value in detail_values), "one": sum(value == 1 for value in detail_values)}
    for name, lower, upper in (("q1", 0, .25), ("q2", .25, .5), ("q3", .5, .75), ("q4", .75, 1)):
        histogram[name] = sum(lower < value <= upper and value < 1 for value in detail_values)
    ctx.save_json("dev_frontier.json", frontier)
    choice = results["saver"]["f10"]["c25"]
    low = auc["saver"]["post"]["ci_low"]
    discrimination = bool(choice["better_random_harm"] or choice["better_random_dev_f1"] or (low is not None and low > .5))
    c2_pass = bool(choice["feasible_nonempty"] and 0 < choice["dev"]["detail"] < 1 and discrimination)
    delta1 = json_read("repro/p24/phase5_delta1.json")["delta1"]
    auc_point = auc["saver"]["post"]["auroc"]
    uncertainty_point = auc["uncertainty"]["post"]["auroc"]
    decision = {"fallback_comparison": fallback, "fallback_pass": int(fallback["ci_low"] >= -delta1),
                "forced_on_comparison": forced, "nontrivial_thresholds": nontrivial_thresholds,
                "nonsaturation_pass": int(nontrivial_thresholds >= 10), "discrimination_pass": int(discrimination),
                "c2_seed_pass": int(c2_pass), "auroc_success": int(low is not None and low > .5 and auc_point is not None
                                                                          and uncertainty_point is not None and auc_point > uncertainty_point)}
    ctx.save_json("METHOD_LOCK.json", {"study": study, "locks": locks, "decisions": decision, "selection_splits": ["dev", "calibration"]})
    if study == "legacy":
        audit, _ = find_run(sources["phase2_root"], ctx.seed, experiment="phase2_audit", extra_tags={"parent_run_id": trained.name})
        branches["saver"]["test"] = final_epoch(audit / "raw/frozen_test.jsonl")
    else:
        branches["saver"]["test"] = final_epoch(trained / "raw/test.jsonl")
    branches["uncertainty"]["test"] = branches["saver"]["test"]
    if "maxsim" in branches:
        branches["maxsim"]["test"] = final_epoch(branches["maxsim"]["test_path"])
    anchors["test"] = final_epoch(text / "raw/test.jsonl")
    score_test(results, locks, branches, anchors, manifest)
    comparison_grid = {}
    for fraction in results["saver"]:
        if "status" in results["saver"][fraction] or "status" in results["uncertainty"][fraction]:
            continue
        comparison_grid[fraction] = {}
        for cap in ("c25", "c35", "c50"):
            learned = results["saver"][fraction][cap]
            baseline = results["uncertainty"][fraction][cap]
            comparison_grid[fraction][cap] = {"uncertainty_dev_f1_delta": baseline["dev"]["f1"] - learned["dev"]["f1"],
                                               "uncertainty_calibration_harm_delta": baseline["calibration"]["harm"] - learned["calibration"]["harm"],
                                               "uncertainty_no_worse": int(baseline["dev"]["f1"] >= learned["dev"]["f1"] - 1e-12 and
                                                                           baseline["calibration"]["harm"] <= learned["calibration"]["harm"] + 1e-12)}
    label_rates = {}
    for split, rows in branches["saver"].items():
        labels = [row.get("utility_target", [None])[0] for row in rows]
        known = [value for value in labels if value is not None]
        label_rates[split] = float(np.mean(known)) if known else None
    ctx.save_json("certificate_replay.json", results)
    ctx.save_json("auroc.json", auc)
    ctx.save_json("decision.json", decision)
    ctx.set_metrics({"certificates": results, "auroc": auc, "decision": decision, "label_positive_rates": label_rates,
                     "dev_threshold_histogram": histogram, "selected_gate_mean": json_read(trained / "best.json").get("gate_mean"),
                     "uncertainty_comparison": comparison_grid,
                     "training_updates": 0, "gpu_hours_actual": 0., "new_test_forward_calls": 0})
