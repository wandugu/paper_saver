"""Numeric phase4 tables; values are read only by repro.make_tables."""
from repro.common.tables import ColumnSpec, RowSpec, TableSpec

RUNS = {42: "P24_phase4_audit_s42_20261113T161724Z_7311c6", 43: "P24_phase4_audit_s43_20261113T161731Z_8066cd", 44: "P24_phase4_audit_s44_20261113T161737Z_c072d8"}
TABLES = []

def add(name, caption, rows, columns):
    TABLES.append(TableSpec(name, caption, rows, [ColumnSpec(*column) for column in columns]))

def row_seed(seed):
    return RowSpec(f"seed{seed}", {"run_id": RUNS[seed], "experiment": "phase4_audit", "status": "SUCCESS"})

def row_method(seed, method):
    return RowSpec(f"seed{seed}", {"run_id": RUNS[seed], "experiment": "phase4_audit", "status": "SUCCESS"})

for cap in ("c25", "c35", "c50"):
    prefix = f"d1.{cap}."
    add(f"p24_phase4_D1_dual_{cap}", f"D1 dual certificate, cap={cap[1:]}%; per-seed locked choice.",
        [row_seed(seed) for seed in (42, 43, 44)],
        [("tau", prefix + "threshold", "{:.2f}", "single"), ("Harm UCB", prefix + "harm_ucb", "{:.5f}", "single"),
         ("Cost UCB", prefix + "cost_ucb", "{:.5f}", "single"), ("Test F1", prefix + "test.f1", "{:.5f}", "single"),
         ("Test harm", prefix + "test.harm", "{:.5f}", "single"), ("D/post", prefix + "test.detail", "{:.5f}", "single"),
         ("Norm cost", prefix + "test.cost", "{:.5f}", "single"), ("Replay ms", prefix + "test.latency_ms", "{:.3f}", "single"),
         ("F1-text", prefix + "f1_delta_vs_text", "{:.5f}", "single"), ("F1 regret", prefix + "f1_regret", "{:.5f}", "single"),
         ("Fallback", prefix + "fallback", "{:.0f}", "single"), ("F1>=text", prefix + "test_f1_ge_text", "{:.0f}", "single")])
    for selector in ("harm_only", "max_dev_f1", "cost_penalty"):
        sp = f"{prefix}{selector}."
        add(f"p24_phase4_D1_{selector}_{cap}", f"D1 {selector} reference, cap={cap[1:]}%; harm-only comparison may violate cost cap.",
            [row_seed(seed) for seed in (42, 43, 44)],
            [("tau", sp + "threshold", "{:.2f}", "single"), ("Harm UCB", sp + "harm_ucb", "{:.5f}", "single"),
             ("Cost UCB", sp + "cost_ucb", "{:.5f}", "single"), ("Test F1", sp + "test.f1", "{:.5f}", "single"),
             ("Norm cost", sp + "test.cost", "{:.5f}", "single"), ("F1-text", sp + "f1_delta_vs_text", "{:.5f}", "single"),
             ("Fallback", sp + "fallback", "{:.0f}", "single")])

for method in ("saver", "maxsim", "linear", "utility_mlp_k1"):
    for cap in ("c25", "c35", "c50"):
        prefix = f"d2.{method}.{cap}."
        add(f"p24_phase4_D2_{method}_{cap}", f"D2 {method} dual selector, cap={cap[1:]}%; same certificate and frozen logits.",
            [row_seed(seed) for seed in (42, 43, 44)],
            [("tau", prefix + "threshold", "{:.2f}", "single"), ("Harm UCB", prefix + "harm_ucb", "{:.5f}", "single"),
             ("Cost UCB", prefix + "cost_ucb", "{:.5f}", "single"), ("Test F1", prefix + "test.f1", "{:.5f}", "single"),
             ("Test harm", prefix + "test.harm", "{:.5f}", "single"), ("Norm cost", prefix + "test.cost", "{:.5f}", "single"),
             ("F1-text", prefix + "f1_delta_vs_text", "{:.5f}", "single"), ("Fallback", prefix + "fallback", "{:.0f}", "single")])

for fraction in (2, 5, 10, 20):
    prefix = f"d3.saver.c{fraction:02d}."
    add(f"p24_phase4_D3_saver_c{fraction:02d}", f"D3 SAVER dual feasibility/fallback at {fraction}% calibration; empirical per-seed frequencies.",
        [row_seed(seed) for seed in (42, 43, 44)],
        sum(([ (f"{cap} feasible", prefix + f"{cap}.nonempty_feasible", "{:.0f}", "single"),
              (f"{cap} fallback", prefix + f"{cap}.fallback", "{:.0f}", "single"),
              (f"{cap} feasible policies", prefix + f"{cap}.feasible_policies", "{:.0f}", "single")] for cap in ("c25", "c35", "c50")), []))

for method in ("maxsim", "linear", "utility_mlp_k1"):
    prefix = f"d3.{method}.c10."
    add(f"p24_phase4_D3_{method}_c10", f"D3 {method} dual feasibility/fallback at 10% calibration.",
        [row_seed(seed) for seed in (42, 43, 44)],
        [(f"{cap} feasible", prefix + f"{cap}.nonempty_feasible", "{:.0f}", "single") for cap in ("c25", "c35", "c50")] +
        [(f"{cap} fallback", prefix + f"{cap}.fallback", "{:.0f}", "single") for cap in ("c25", "c35", "c50")])

for cap in ("c25", "c35", "c50"):
    prefix = f"random_gate.{cap}."
    add(f"p24_phase4_random_{cap}", f"D2 rate-matched Random Gate, cap={cap[1:]}%; five deterministic seeds.",
        [row_seed(seed) for seed in (42, 43, 44)],
        [("Matched rate", prefix + "target_activation_rate", "{:.5f}", "single"), ("F1 mean", prefix + "f1_mean", "{:.5f}", "single"),
         ("F1 min", prefix + "f1_min", "{:.5f}", "single"), ("F1 max", prefix + "f1_max", "{:.5f}", "single"),
         ("Harm mean", prefix + "harm_mean", "{:.5f}", "single"), ("Cost mean", prefix + "cost_mean", "{:.5f}", "single")])
