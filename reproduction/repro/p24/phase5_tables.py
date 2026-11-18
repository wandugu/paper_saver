"""Tables from phase5 runs, including all preregistered stop decisions."""

import os
import json
from functools import lru_cache
from pathlib import Path

from repro.common.tables import ColumnSpec, RowSpec, TableSpec
from repro.p24.protocol import read_records


ROOT = Path(os.environ.get("P24_PHASE5_RUNS", Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase5"))
LEDGER = read_records(ROOT / "ledger.jsonl")
TABLES = []


def add(name, caption, rows, columns):
    if rows:
        TABLES.append(TableSpec(name, caption, rows, [ColumnSpec(*column) for column in columns]))


@lru_cache(maxsize=None)
def metrics_for(run_id):
    source = next(row for row in LEDGER if row["run_id"] == run_id)
    return json.loads((ROOT / source["run_dir"] / "metrics.json").read_text())


def random_rows(rows, scorer, fraction, cap):
    return [row for row in rows if metrics_for(row.select["run_id"]).get("certificates", {}).get(scorer, {}).get(fraction, {}).get(cap, {}).get("random_dev") is not None]


def selected(experiment, tags):
    matches = [row for row in LEDGER if row["status"] == "SUCCESS" and row["experiment"] == experiment
               and all(row.get("tags", {}).get(key) == value for key, value in tags.items())]
    return [RowSpec(f"seed{row['seed']}", {"run_id": row["run_id"], "status": "SUCCESS"}) for row in sorted(matches, key=lambda value: value["seed"])]


rows = selected("phase5_checkpoint", {"checkpoint": "C1"})
if rows:
    add("p24_phase5_C1", "Post hoc C1: paired post bootstrap, exactly original 512 dev units; fractions.", rows,
        [("All-off F1", "all_off_dev_f1", "{:.5f}", "single"), ("F1 delta", "comparison.f1_delta", "{:.5f}", "single"),
         ("CI low", "comparison.ci_low", "{:.5f}", "single"), ("CI high", "comparison.ci_high", "{:.5f}", "single"),
         ("delta1", "delta1", "{:.5f}", "single"), ("Pass", "pass", "{:.0f}", "single")])

for study in ("A1", "A2", "A6", "A7", "A7_maxsim"):
    methods = ("text_only", "always_vision") if study == "A6" else ("maxsim",) if study == "A7_maxsim" else ("saver",)
    for method in methods:
        rows = selected("phase5_training", {"study": study, "method": method})
        if rows:
            add(f"p24_phase5_{study}_{method}", f"Post hoc {study} {method}; test descriptive only.", rows,
                [("Dev F1", "development_f1", "{:.5f}", "single"), ("All-off F1", "all_off_dev_f1", "{:.5f}", "single"),
                 ("Test F1", "test_f1", "{:.5f}", "single"), ("Fit units", "fit_units", "{:.0f}", "single"),
                 ("Dev units", "dev_units", "{:.0f}", "single"), ("Seconds", "seconds", "{:.1f}", "single")])

for study in ("A1", "A2", "legacy", "A7"):
    rows = selected("phase5_analysis", {"study": study, "analysis_revision": "v4"}) or selected("phase5_analysis", {"study": study, "analysis_revision": "v3"})
    if not rows:
        continue
    add(f"p24_phase5_{study}_decisions", f"Post hoc {study} dev/calibration decisions; test excluded.", rows,
        [("Fallback delta", "decision.fallback_comparison.f1_delta", "{:.5f}", "single"),
         ("CI low", "decision.fallback_comparison.ci_low", "{:.5f}", "single"),
         ("Fallback pass", "decision.fallback_pass", "{:.0f}", "single"),
         ("Nontrivial thresholds", "decision.nontrivial_thresholds", "{:.0f}", "single"),
         ("On delta", "decision.forced_on_comparison.f1_delta", "{:.5f}", "single"),
         ("On CI low", "decision.forced_on_comparison.ci_low", "{:.5f}", "single"),
         ("C2 seed pass", "decision.c2_seed_pass", "{:.0f}", "single")])
    add(f"p24_phase5_{study}_threshold_distribution", "All 101 dev thresholds: D/post histogram; complete per-threshold predictions in dev_frontier.json; selected checkpoint gate mean.", rows,
        [("D=0", "dev_threshold_histogram.zero", "{:.0f}", "single"),
         ("0<D<=.25", "dev_threshold_histogram.q1", "{:.0f}", "single"),
         (".25<D<=.5", "dev_threshold_histogram.q2", "{:.0f}", "single"),
         (".5<D<=.75", "dev_threshold_histogram.q3", "{:.0f}", "single"),
         (".75<D<1", "dev_threshold_histogram.q4", "{:.0f}", "single"),
         ("D=1", "dev_threshold_histogram.one", "{:.0f}", "single"),
         ("Selected gate mean", "selected_gate_mean", "{:.5f}", "single")])
    if study != "legacy":
        add(f"p24_phase5_{study}_CE_margin_label_rates", "CE-margin target positive rates, distinct from the new correctness-gain AUROC labels; test descriptive.", rows,
            [("Dev CE-positive", "label_positive_rates.dev", "{:.5f}", "single"),
             ("Cal CE-positive", "label_positive_rates.calibration", "{:.5f}", "single"),
             ("Test CE-positive", "label_positive_rates.test", "{:.5f}", "single")])
    for scorer in ("saver", "maxsim", "uncertainty"):
        for level in ("unit", "post"):
            prefix = f"auroc.{scorer}.{level}."
            add(f"p24_phase5_{study}_A3a_{scorer}_{level}", f"Post hoc {study} calibration anchor-gain {level} AUROC; zero gains are negative labels.", rows,
                [("AUROC", prefix + "auroc", "{:.5f}", "single"), ("CI low", prefix + "ci_low", "{:.5f}", "single"),
                 ("CI high", prefix + "ci_high", "{:.5f}", "single"), ("Brier", prefix + "brier", "{:.5f}", "single"),
                 ("Positive", prefix + "positive", "{:.0f}", "single"), ("Negative gain", prefix + "negative", "{:.0f}", "single"),
                 ("Zero gain", prefix + "zero", "{:.0f}", "single"), ("n", prefix + "n", "{:.0f}", "single")])
        for fraction in ("f02", "f05", "f10", "f20"):
            for endpoint in ("off", "on"):
                prefix = f"certificates.{scorer}.{fraction}.endpoints.{endpoint}."
                add(f"p24_phase5_{study}_A4_{scorer}_{fraction}_{endpoint}", f"Post hoc {study} {scorer} {fraction} {endpoint} endpoint; no test selection.", rows,
                    [("Cal F1", prefix + "calibration.f1", "{:.5f}", "single"), ("Cal harm", prefix + "calibration.harm", "{:.5f}", "single"),
                     ("Cal cost", prefix + "calibration.cost", "{:.5f}", "single"), ("Dev F1", prefix + "dev.f1", "{:.5f}", "single"),
                     ("Test F1", prefix + "test.f1", "{:.5f}", "single"), ("Test D/post", prefix + "test.detail", "{:.5f}", "single")])
            for cap in ("c25", "c35", "c50"):
                prefix = f"certificates.{scorer}.{fraction}.{cap}."
                add(f"p24_phase5_{study}_A4_{scorer}_{fraction}_{cap}", f"Post hoc {study} {scorer}, {fraction}, {cap}; missing means not applicable; test descriptive.", rows,
                    [("tau", prefix + "threshold", "{:.2f}", "single"), ("Harm UCB", prefix + "harm_ucb", "{:.5f}", "single"),
                     ("Cost UCB", prefix + "cost_ucb", "{:.5f}", "single"), ("Dev F1", prefix + "dev.f1", "{:.5f}", "single"),
                     ("Dev D/post", prefix + "dev.detail", "{:.5f}", "single"), ("Test F1", prefix + "test.f1", "{:.5f}", "single"),
                     ("Test D/post", prefix + "test.detail", "{:.5f}", "single"), ("F1-text", prefix + "test_f1_delta_anchor", "{:.5f}", "single"),
                     ("Fallback", prefix + "fallback", "{:.0f}", "single"), ("Feasible", prefix + "feasible_policies", "{:.0f}", "single")])
                eligible_rows = random_rows(rows, scorer, fraction, cap)
                add(f"p24_phase5_{study}_A9_{scorer}_{fraction}_{cap}", f"Post hoc exact-rate Random controls for {study} {scorer} {fraction} {cap}; five seeds; structural all-off cells omitted, historical zero-rate rows retained.", eligible_rows,
                    [("Dev rate", prefix + "random_dev.target_rate", "{:.5f}", "single"), ("Dev F1 mean", prefix + "random_dev.f1_mean", "{:.5f}", "single"),
                     ("Cal rate", prefix + "random_calibration.target_rate", "{:.5f}", "single"), ("Cal harm mean", prefix + "random_calibration.harm_mean", "{:.5f}", "single"),
                     ("Test rate", prefix + "random_test.target_rate", "{:.5f}", "single"), ("Test F1 mean", prefix + "random_test.f1_mean", "{:.5f}", "single"),
                     ("Test F1 min", prefix + "random_test.f1_min", "{:.5f}", "single"), ("Test F1 max", prefix + "random_test.f1_max", "{:.5f}", "single")])
                if scorer == "saver":
                    comparison_prefix = f"uncertainty_comparison.{fraction}.{cap}."
                    add(f"p24_phase5_{study}_A4_gate_value_{fraction}_{cap}",
                        "Text uncertainty minus learned gate; no_worse=1 means the learned gate adds no value in this cell.", rows,
                        [("Text Dev F1 delta", comparison_prefix + "uncertainty_dev_f1_delta", "{:.5f}", "single"),
                         ("Text Cal harm delta", comparison_prefix + "uncertainty_calibration_harm_delta", "{:.5f}", "single"),
                         ("Text no worse", comparison_prefix + "uncertainty_no_worse", "{:.0f}", "single")])
                for split in ("calibration", "dev", "test"):
                    random_prefix = prefix + f"random_{split}."
                    columns = [("Target unit rate", random_prefix + "target_rate", "{:.5f}", "single")]
                    for random_seed in (2401, 2402, 2403, 2404, 2405):
                        for title, metric in (("rate", "coverage"), ("D/post", "detail"), ("F1", "f1"), ("harm", "harm")):
                            columns.append((f"r{random_seed} {title}", random_prefix + f"draws.s{random_seed}.{metric}", "{:.5f}", "single"))
                    for metric in ("f1", "harm", "detail", "coverage"):
                        for statistic in ("mean", "min", "max"):
                            columns.append((f"{metric} {statistic}", random_prefix + f"summary.{metric}.{statistic}", "{:.5f}", "single"))
                    add(f"p24_phase5_{study}_A9_draws_{scorer}_{fraction}_{cap}_{split}",
                        f"All five predeclared Random draws, exact unit-rate matching; {study}/{scorer}/{fraction}/{cap}/{split}; no label-based selection.", eligible_rows, columns)

rows = selected("phase5_checkpoint", {"checkpoint": "C2"})
if rows:
    add("p24_phase5_C2", "Post hoc fixed C2 across all registered seeds; unrun A1 seeds count as unavailable, not successes.", rows,
        [("C2 pass", "pass", "{:.0f}", "single"), ("A1 passing seeds", "variants.A1.c2_passing_seeds", "{:.0f}", "single"),
         ("A2 passing seeds", "variants.A2.c2_passing_seeds", "{:.0f}", "single"),
         ("A1 AUROC mean", "variants.A1.mean_calibration_post_auroc", "{:.5f}", "single"),
         ("A2 AUROC mean", "variants.A2.mean_calibration_post_auroc", "{:.5f}", "single"),
         ("A1 A3a pass", "variants.A1.A3a_success", "{:.0f}", "single"),
         ("A2 A3a pass", "variants.A2.A3a_success", "{:.0f}", "single")])

rows = selected("phase5_checkpoint", {"checkpoint": "C3"})
if rows:
    add("p24_phase5_C3", "Post hoc fixed uncapped A6 go/no-go; dev also used for early stopping, test excluded.", rows,
        [("Go", "go", "{:.0f}", "single"), ("Positive seeds", "positive_seeds", "{:.0f}", "single"),
         ("CI-positive seeds", "positive_ci_seeds", "{:.0f}", "single")])
    for seed in (42, 43, 44):
        prefix = f"seeds.s{seed}."
        paired_rows = [RowSpec(f"seed{seed}", row.select) for row in rows]
        add(f"p24_phase5_A6_paired_s{seed}", f"Post hoc uncapped always-vision minus text-only paired post bootstrap, seed{seed}.", paired_rows,
            [("Text dev F1", prefix + "text_dev_f1", "{:.5f}", "single"), ("Vision dev F1", prefix + "vision_dev_f1", "{:.5f}", "single"),
             ("Delta", prefix + "f1_delta", "{:.5f}", "single"), ("CI low", prefix + "ci_low", "{:.5f}", "single"),
             ("CI high", prefix + "ci_high", "{:.5f}", "single"), ("Dev posts", prefix + "n_posts", "{:.0f}", "single")])

rows = selected("phase5_checkpoint", {"checkpoint": "C4"})
if rows:
    add("p24_phase5_C4", "Post hoc uncapped s42 pilot stop-loss.", rows,
        [("Pass", "pass", "{:.0f}", "single"), ("Dev D/post", "dev_detail", "{:.5f}", "single"),
         ("All-off CI low", "decision.fallback_comparison.ci_low", "{:.5f}", "single")])

rows = selected("phase5_audit", {"study": "consistency_audit"})[-1:]
if rows:
    add("p24_phase5_audit", "Formal pinned-runtime consistency audit; combined cache bytes include excluded attempts.", rows,
        [("Pass", "audit_pass", "{:.0f}", "single"), ("Findings", "finding_count", "{:.0f}", "single"),
         ("Certificate cells", "certificate_cells_checked", "{:.0f}", "single"),
         ("Training runs", "training_runs_checked", "{:.0f}", "single"),
         ("New cache bytes", "total_new_cache_bytes", "{:.0f}", "single"),
         ("Existing cache checked", "unchanged_existing_cache_files", "{:.0f}", "single")])

rows = selected("phase5_diagnostics", {"study": "A5", "diagnostic_revision": "v2"})
if rows:
    add("p24_phase5_A5_oracles", "Post hoc descriptive test oracle; correct-if-either with anchor tie-break, no deployment selection.", rows,
        [("Text F1", "text_f1", "{:.5f}", "single"), ("On F1", "forced_on_f1", "{:.5f}", "single"),
         ("Text/on oracle", "text_vision_oracle_f1", "{:.5f}", "single"),
         ("MaxSim flips", "maxsim_flips.prediction_flips", "{:.0f}", "single"),
         ("MaxSim on-off delta", "maxsim_flips.forced_on_vs_own_off.f1_delta", "{:.5f}", "single")])
    for quintile in range(1, 6):
        prefix = f"confidence_quintiles.q{quintile}."
        add(f"p24_phase5_A5_q{quintile}", f"Post hoc test text-confidence quintile {quintile}; fix/break versus independent anchor.", rows,
            [("n", prefix + "n", "{:.0f}", "single"), ("Fix", prefix + "fix", "{:.0f}", "single"),
             ("Break", prefix + "break", "{:.0f}", "single"), ("Fix-break", prefix + "fix_minus_break", "{:.0f}", "single"),
             ("Fix-break rate", prefix + "fix_minus_break_rate", "{:.5f}", "single")])
    for label in range(23):
        prefix = f"relation_strata.r{label:02d}."
        add(f"p24_phase5_A5_relation_{label:02d}", f"Post hoc test relation id {label}; label mapping retained in source identities.", rows,
            [("n", prefix + "n", "{:.0f}", "single"), ("Fix", prefix + "fix", "{:.0f}", "single"),
             ("Break", prefix + "break", "{:.0f}", "single"), ("Fix-break", prefix + "fix_minus_break", "{:.0f}", "single")])
    for source in LEDGER:
        if source["status"] == "SUCCESS" and source["experiment"] == "phase5_diagnostics" and source.get("tags", {}).get("diagnostic_revision") == "v2":
            current = source["seed"]
            for alternative in (42, 43, 44):
                if alternative == current:
                    continue
                prefix = f"seed_controls.s{alternative}."
                add(f"p24_phase5_A5_seed_{current}_{alternative}", f"Post hoc text seed-control oracle {current} versus {alternative}; test descriptive.",
                    [RowSpec(f"seed{current}", {"run_id": source["run_id"], "status": "SUCCESS"})],
                    [("Anchor F1", "text_f1", "{:.5f}", "single"), ("Alternative F1", prefix + "alternative_f1", "{:.5f}", "single"),
                     ("Oracle F1", prefix + "oracle_f1", "{:.5f}", "single")])
