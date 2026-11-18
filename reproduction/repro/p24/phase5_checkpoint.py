"""Record phase5 dependency decisions through the experiment ledger."""

from pathlib import Path

from .phase3_replay import Population, json_read, paired
from .protocol import read_records
from .phase4_replay import find_run


def main(ctx, config):
    ctx.tags.update(stage="phase5", checkpoint=config["checkpoint"], post_hoc=True, device="cpu")
    if config["checkpoint"] == "C2":
        return checkpoint_c2(ctx, config)
    if config["checkpoint"] == "C3":
        return checkpoint_c3(ctx, config)
    if config["checkpoint"] == "C4":
        return checkpoint_c4(ctx, config)
    if config["checkpoint"] != "C1":
        raise ValueError("Unsupported checkpoint")
    run = Path(config["run"])
    source = json_read(run / "config.json")
    rows = read_records(run / "raw/dev.jsonl")
    anchor = {row["id"]: row for row in read_records(config["anchor_dev"])}
    population = Population(rows, read_records(source["manifest"]))
    comparison = paired(population, population.text, [anchor[key]["prediction"][0] for key in population.ids], ctx.seed)
    delta1 = json_read("repro/p24/phase5_delta1.json")["delta1"]
    passed = comparison["ci_low"] >= -delta1
    result = {"comparison": comparison, "delta1": delta1, "pass": int(passed),
              "all_off_dev_f1": json_read(run / "best.json")["all_off_dev_f1"],
              "decision": "continue A1 s43/s44 and A2" if passed else "skip A1 s43/s44, run A2 three seeds"}
    ctx.save_json("decision.json", result)
    ctx.set_metrics(result)


def checkpoint_c2(ctx, config):
    ledger = read_records(Path(config["phase5_root"]) / "ledger.jsonl")
    variants = {}
    for variant in ("A1", "A2"):
        results = {}
        for row in ledger:
            if row["status"] == "SUCCESS" and row["experiment"] == "phase5_analysis" and row.get("tags", {}).get("analysis_revision") == "v3" and row.get("tags", {}).get("study") == variant:
                seed = str(row["seed"])
                if seed in results:
                    raise ValueError(f"Repeated valid analysis for {variant}/{seed}")
                metrics = json_read(Path(config["phase5_root"]) / row["run_dir"] / "metrics.json")
                results[seed] = {"run_id": row["run_id"], "decision": metrics["decision"], "auroc": metrics["auroc"]["saver"]["post"]}
        values = [row["auroc"]["auroc"] for row in results.values() if row["auroc"]["auroc"] is not None]
        passed = sum(row["decision"]["c2_seed_pass"] for row in results.values())
        variants[variant] = {"seeds": results, "c2_passing_seeds": passed,
                             "mean_calibration_post_auroc": sum(values) / len(values) if values else None,
                             "A3a_success": int(len(results) == 3 and all(row["decision"]["auroc_success"] for row in results.values()))}
        variants[variant]["stop_router_variants"] = int(len(results) == 3 and
            all(row["decision"]["forced_on_comparison"]["ci_low"] <= 0 for row in results.values()) and
            all(row["auroc"]["ci_low"] is not None and row["auroc"]["ci_low"] <= .5 <= row["auroc"]["ci_high"] for row in results.values()))
    selected = max(variants, key=lambda name: (variants[name]["c2_passing_seeds"],
                                              variants[name]["mean_calibration_post_auroc"] if variants[name]["mean_calibration_post_auroc"] is not None else -1,
                                              int(name == "A2")))
    passed = variants[selected]["c2_passing_seeds"] >= 2
    result = {"pass": int(passed), "variants": variants, "selected_configuration": selected,
              "decision": "allow A7 s42 alongside A6" if passed else "skip A7 and A8; run A6 as analytical evidence",
              "selection_splits": ["dev", "calibration"]}
    ctx.save_json("decision.json", result)
    ctx.set_metrics(result)


def checkpoint_c3(ctx, config):
    comparisons = {}
    for seed in (42, 43, 44):
        parents = {}
        for method in ("text_only", "always_vision"):
            directory, row = find_run(config["phase5_root"], seed, experiment="phase5_training", tag_key="study", tag_value="A6", extra_tags={"method": method})
            parents[method] = (directory, row)
        text_rows = sorted(read_records(parents["text_only"][0] / "raw/dev.jsonl"), key=lambda row: row["id"])
        vision_rows = sorted(read_records(parents["always_vision"][0] / "raw/dev.jsonl"), key=lambda row: row["id"])
        if [row["id"] for row in text_rows] != [row["id"] for row in vision_rows]:
            raise ValueError("A6 dev IDs differ")
        if len(text_rows) != 1624:
            raise ValueError("A6 dev is not full")
        settings = json_read(parents["text_only"][0] / "config.json")
        population = Population(text_rows, read_records(settings["manifest"]))
        comparison = paired(population, [row["prediction"][0] for row in vision_rows], [row["prediction"][0] for row in text_rows], seed)
        comparisons[f"s{seed}"] = {**comparison, "text_run_id": parents["text_only"][1]["run_id"],
                                    "vision_run_id": parents["always_vision"][1]["run_id"],
                                    "text_dev_f1": json_read(parents["text_only"][0] / "metrics.json")["development_f1"],
                                    "vision_dev_f1": json_read(parents["always_vision"][0] / "metrics.json")["development_f1"]}
    positive = sum(row["f1_delta"] > 0 for row in comparisons.values())
    significant = sum(row["ci_low"] > 0 for row in comparisons.values())
    go = positive == 3 and significant >= 2
    result = {"go": int(go), "positive_seeds": positive, "positive_ci_seeds": significant, "seeds": comparisons,
              "decision": "continue preregistered A7 if C2/C4 pass and before launch deadline" if go else "no-go; stop accuracy-oriented GPU work including running A7, and do not start remaining A7",
              "dev_selection_bias": "Full dev was also used for checkpoint early stopping in both arms."}
    ctx.save_json("decision.json", result)
    ctx.set_metrics(result)


def checkpoint_c4(ctx, config):
    directory, row = find_run(config["phase5_root"], 42, experiment="phase5_analysis", tag_key="study", tag_value="A7", extra_tags={"analysis_revision": "v3"})
    metrics = json_read(directory / "metrics.json")
    choice = metrics["certificates"]["saver"]["f10"]["c25"]
    passed = metrics["decision"]["fallback_pass"] and 0 < choice["dev"]["detail"] < 1
    result = {"pass": int(passed), "pilot_run_id": row["run_id"], "decision": metrics["decision"],
              "dev_detail": choice["dev"]["detail"], "continue_remaining": int(passed)}
    ctx.save_json("decision.json", result)
    ctx.set_metrics(result)
