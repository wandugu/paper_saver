"""Consistency audit of phase5 splits, anchors, locked choices and evidence."""

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

from ..common.paths import require_env
from .phase3_replay import json_read
from .phase4_replay import find_run
from .phase5_preregister import compute
from .phase5_training import FROZEN_MODULES, excluded_cache_entries
from .protocol import digest, post_key, read_records, split_records, task_f1


def main(ctx, config):
    if torch.cuda.is_available():
        raise RuntimeError("Consistency audit is CPU only")
    ctx.tags.update(stage="phase5", study="consistency_audit", device="cpu", post_hoc=True)
    root = Path(config["phase5_root"])
    sources = json_read(config["sources"])
    ledger = read_records(root / "ledger.jsonl")
    manifest_rows = read_records(sources["manifest"])
    manifest = {row["id"]: row for row in manifest_rows}
    findings, checks = [], {}
    original_delta = json_read("repro/p24/phase5_delta1.json")
    recomputed_delta = compute(sources)
    if recomputed_delta != original_delta:
        findings.append("Preregistered dev tolerance does not reproduce")
    checks["delta1_reproduced"] = recomputed_delta["delta1"] == original_delta["delta1"]
    exclusions = json_read("repro/p24/phase5_runtime_exclusions.json")
    for row in ledger:
        if row.get("status") != "SUCCESS":
            continue
        if subprocess.run(["git", "merge-base", "--is-ancestor", "aa7c5c76045ce8ee96ecc4bd0e7cbcde1ba08cf0", row["git_commit"]], capture_output=True).returncode:
            findings.append(f"Run does not descend from prereg: {row['run_id']}")
        if row.get("git_dirty"):
            findings.append(f"Dirty code in run: {row['run_id']}")
        if row["run_id"] in exclusions["runs"]:
            findings.append(f"Excluded runtime run entered formal ledger: {row['run_id']}")
        if subprocess.run(["git", "merge-base", "--is-ancestor", "51296aabe3336733811acb78076ce6acf31812af", row["git_commit"]], capture_output=True).returncode:
            findings.append(f"Run predates pinned runtime amendment: {row['run_id']}")
        environment = json_read(root / row["run_dir"] / "env.json")
        pinned_python = require_env("P24_REPO_ROOT") + "/.venv-p24-310/bin/python"
        if environment["python_executable"] != pinned_python or environment["versions"]["transformers"] != "4.57.1" or environment["versions"]["torch"] != "2.6.0+cu118":
            findings.append(f"Formal runtime mismatch: {row['run_id']}")
        if row["experiment"] in ("phase5_training", "phase5_cache"):
            pinned_gpu_uuid = require_env("P24_PHASE5_GPU_UUID")
            hardware = json_read(root / row["run_dir"] / "gpu_initial.json")
            if environment["cuda_visible_devices"] != "1" or hardware["physical_gpu"] != 1 or hardware["uuid"] != pinned_gpu_uuid:
                findings.append(f"GPU assignment differs from locked GPU1: {row['run_id']}")
    checks["successful_runs_prereg_ancestors_checked"] = sum(row["status"] == "SUCCESS" for row in ledger)
    training = [row for row in ledger if row["status"] == "SUCCESS" and row["experiment"] == "phase5_training"]
    for row in training:
        directory = root / row["run_dir"]
        settings = json_read(directory / "config.json")
        identity = json_read(directory / "data_identity.json")
        splits = identity["split_ids"]
        expected_splits = split_records(manifest_rows, row["seed"], fit_cap=settings.get("fit_cap", 4096), dev_cap=settings.get("dev_cap", 512))
        if splits != {name: [record["id"] for record in records] for name, records in expected_splits.items()}:
            findings.append(f"Saved split differs from fixed seed protocol: {row['run_id']}")
        if identity["sha256"] != digest(sources["manifest"]):
            findings.append(f"Training manifest identity differs: {row['run_id']}")
        fit = {post_key(manifest[key]) for key in splits["fit"]}
        calibration = {post_key(manifest[key]) for key in splits["calibration"]}
        if fit & calibration:
            findings.append(f"Fit/calibration post overlap: {row['run_id']}")
        study = row["tags"]["study"]
        if study == "A6" and (len(splits["dev"]) != 1624 or settings["fit_cap"] is not None or settings["dev_cap"] is not None):
            findings.append(f"A6 caps not removed: {row['run_id']}")
        if study == "A6" and len(splits["fit"]) != {42: 10969, 43: 11014, 44: 10989}[row["seed"]]:
            findings.append(f"A6 full fit size differs: {row['run_id']}")
        dev_rows = read_records(directory / "raw/dev.jsonl")
        metrics = json_read(directory / "metrics.json")
        if abs(task_f1(dev_rows, "mre") - metrics["development_f1"]) > 1e-12:
            findings.append(f"Dev F1 differs from saved predictions: {row['run_id']}")
        if study == "A2" or (study == "A7" and settings.get("variant") == "A2"):
            anchor = torch.load(settings["anchor_checkpoint"], map_location="cpu", weights_only=True)
            trained = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            frozen = [name for name in anchor if name.split(".")[0] in FROZEN_MODULES]
            if any(not torch.equal(anchor[name], trained[name]) for name in frozen):
                findings.append(f"Trained A2 anchor changed: {row['run_id']}")
    checks["training_runs_checked"] = len(training)
    checks["A2_seeds"] = sorted(row["seed"] for row in training if row["tags"]["study"] == "A2")
    checks["A6_seeds_methods"] = sorted([row["seed"], row["tags"]["method"]] for row in training if row["tags"]["study"] == "A6")
    if checks["A2_seeds"] != [42, 43, 44] or len(checks["A6_seeds_methods"]) != 6:
        findings.append("Mandatory training seed grid incomplete")
    if checks["A6_seeds_methods"] != [[seed, method] for seed in (42, 43, 44) for method in ("always_vision", "text_only")]:
        findings.append("A6 seed/method grid is not the preregistered Cartesian product")
    certificate_cells = 0
    zero_dev_nonfallback = []
    for row in ledger:
        if row["status"] != "SUCCESS" or row["experiment"] != "phase5_analysis" or row.get("tags", {}).get("analysis_revision") not in ("v3", "v4"):
            continue
        directory = root / row["run_dir"]
        metrics = json_read(directory / "metrics.json")
        histogram = metrics["dev_threshold_histogram"]
        if sum(histogram.values()) != 101 or sum(histogram[key] for key in ("q1", "q2", "q3", "q4")) != metrics["decision"]["nontrivial_thresholds"]:
            findings.append(f"Dev threshold distribution incomplete: {row['run_id']}")
        locks = json_read(directory / "METHOD_LOCK.json")["locks"]
        for scorer, fractions in metrics["certificates"].items():
            if "status" in fractions:
                continue
            for fraction_key, caps in fractions.items():
                if "status" in caps:
                    continue
                for cap_key in ("c25", "c35", "c50"):
                    choice = caps[cap_key]
                    cal = choice["calibration"]
                    n, k = cal["n_posts"], cal["harmful_posts"]
                    structural = locks[scorer][fraction_key][cap_key]["threshold"] is None
                    harm = 0. if structural else (1. if k == n else float(beta.ppf(1 - .05 / 202, k + 1, n - k)))
                    cost = 0. if structural else min(1., cal["cost"] + math.sqrt(math.log(202 / .05) / (2 * n)))
                    if abs(harm - choice["harm_ucb"]) > 1e-12 or abs(cost - choice["cost_ucb"]) > 1e-12:
                        findings.append(f"Certificate formula differs: {row['run_id']}/{scorer}/{fraction_key}/{cap_key}")
                    if choice["harm_ucb"] > .1 + 1e-12 or choice["cost_ucb"] > int(cap_key[1:]) / 100 + 1e-12:
                        findings.append(f"Deployed choice violates bound: {row['run_id']}/{scorer}/{fraction_key}/{cap_key}")
                    if choice["dev"]["detail"] == 0 and not choice["fallback"]:
                        zero_dev_nonfallback.append({"run_id": row["run_id"], "scorer": scorer, "fraction": fraction_key, "cap": cap_key})
                    for split in ("random_calibration", "random_dev", "random_test"):
                        random = choice.get(split)
                        if random is None:
                            continue
                        for draw in random["values"]:
                            expected = round(random["target_rate"] * draw["n_units"]) / draw["n_units"]
                            if abs(draw["coverage"] - expected) > 1e-12:
                                findings.append(f"Random activation count mismatch: {row['run_id']}/{split}")
                    certificate_cells += 1
        if row["tags"]["analysis_revision"] == "v4" and row["tags"]["study"] in ("A1", "A2"):
            earlier, _ = find_run(root, row["seed"], experiment="phase5_analysis", tag_key="study", tag_value=row["tags"]["study"], extra_tags={"analysis_revision": "v3"})
            previous = json_read(earlier / "metrics.json")
            for scorer in ("saver", "maxsim", "uncertainty"):
                if previous["certificates"][scorer]["f10"] != metrics["certificates"][scorer]["f10"]:
                    findings.append(f"Extension changed original C2 fraction: {row['run_id']}/{scorer}")
            if previous["decision"] != metrics["decision"]:
                findings.append(f"Extension changed locked decisions: {row['run_id']}")
    cache_run, _ = find_run(root, 42, experiment="phase5_cache", tag_key="study", tag_value="cache_fill")
    before = json_read(cache_run / "cache_before.json")["files"]
    for path, value in before.items():
        if not Path(path).exists() or digest(path) != value["sha256"]:
            findings.append(f"Preexisting feature changed: {path}")
    added_files = {}
    old_root = Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase5"
    for cache_root in (old_root, root):
        for additions_path in cache_root.glob("P24/*/cache_additions.json"):
            for item in json_read(additions_path)["files"]:
                previous = added_files.get(item["path"])
                if previous is not None and previous != item:
                    findings.append(f"Conflicting cache addition identity: {item['path']}")
                added_files[item["path"]] = item
    cache_config = json_read(cache_run / "config.json")
    feature_root = Path(cache_config["cache"])
    if not feature_root.is_absolute():
        feature_root = Path(require_env("P24_REPO_ROOT")) / feature_root
    receipt_roots = {Path(path).parent.parent.resolve() for path in before}
    if receipt_roots != {feature_root.resolve()}:
        raise ValueError("Cache audit receipt does not match the effective feature root")
    invalid_entries = excluded_cache_entries()
    for entry in invalid_entries:
        path = str(feature_root.joinpath(*entry.parts))
        if path not in added_files:
            added_files[path] = {"path": path, "bytes": Path(path).stat().st_size, "sha256": digest(path)}
    for path, item in added_files.items():
        if not Path(path).is_file() or Path(path).stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            findings.append(f"New cache identity changed: {path}")
    cache_bytes = sum(item["bytes"] for item in added_files.values())
    if cache_bytes > 10000000000:
        findings.append("Total cache additions exceed 10GB")
    checks.update(certificate_cells_checked=certificate_cells, zero_dev_nonfallback_cells=zero_dev_nonfallback,
                  total_new_cache_bytes=cache_bytes, total_new_cache_files=len(added_files),
                  excluded_runtime_cache_files=len(invalid_entries),
                  formal_runtime_checked=True, unchanged_existing_cache_files=len(before))
    for checkpoint in ("C1", "C2", "C3"):
        directory, _ = find_run(root, 42, experiment="phase5_checkpoint", tag_key="checkpoint", tag_value=checkpoint)
        checks[checkpoint] = json_read(directory / "decision.json")
    if not checks["C2"]["pass"] and any(row["tags"].get("study") == "A7" for row in training):
        findings.append("A7 run despite C2 failure")
    if checks["C2"]["pass"]:
        pilot = [row for row in training if row["tags"].get("study") == "A7" and row["seed"] == 42]
        if len(pilot) != 1:
            findings.append("C2 passed but the registered A7 pilot is incomplete")
        directory, _ = find_run(root, 42, experiment="phase5_checkpoint", tag_key="checkpoint", tag_value="C4")
        checks["C4"] = json_read(directory / "decision.json")
        remaining = [row for row in training if row["tags"].get("study") == "A7_maxsim" or (row["tags"].get("study") == "A7" and row["seed"] != 42)]
        if not (checks["C3"]["go"] and checks["C4"]["pass"]) and remaining:
            findings.append("Remaining A7 jobs ran despite C3/C4 stop loss")
        for row in training:
            if row["tags"].get("study") not in ("A7", "A7_maxsim"):
                continue
            settings = json_read(root / row["run_dir"] / "config.json")
            if settings["fit_cap"] is not None or settings["dev_cap"] is not None:
                findings.append(f"A7 unexpectedly capped: {row['run_id']}")
            if row["tags"]["study"] == "A7" and settings["variant"] != checks["C2"]["selected_configuration"]:
                findings.append(f"A7 configuration differs from fixed C2 selection: {row['run_id']}")
    expected_a1 = [42, 43, 44] if checks["C1"]["pass"] else [42]
    checks["A1_seeds"] = sorted(row["seed"] for row in training if row["tags"]["study"] == "A1")
    if checks["A1_seeds"] != expected_a1:
        findings.append("A1 seed grid does not respect formal C1 decision")
    ctx.save_json("audit.json", {"findings": findings, "checks": checks,
                  "legacy_debug_exclusions": ["analysis without revision: ID-order error", "analysis v2: before true forced-on MaxSim extraction", "A5 without revision: missing forced-on branch"]})
    ctx.set_metrics({"audit_pass": int(not findings), "finding_count": len(findings),
                     "certificate_cells_checked": certificate_cells, "total_new_cache_bytes": cache_bytes,
                     "unchanged_existing_cache_files": len(before), "training_runs_checked": len(training)})
    if findings:
        raise RuntimeError("Consistency audit failed: " + "; ".join(findings[:10]))
