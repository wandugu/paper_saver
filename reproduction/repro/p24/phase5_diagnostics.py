"""Descriptive A5 oracles, confidence strata, and within-checkpoint flips."""

from pathlib import Path

import numpy as np

from .phase3_replay import json_read, paired
from .phase4_replay import final_epoch, find_run, policies
from .phase5_analysis import align, index, parent, population
from .protocol import read_records


def oracle(population_, alternative):
    prediction = np.where(alternative == population_.gold, alternative, population_.text)
    return population_.metrics(prediction)["f1"]


def strata(population_, confidence, labels):
    correct_text = population_.text == population_.gold
    correct_vision = population_.vision == population_.gold
    result = {}
    for name, indices in labels.items():
        fix = int(np.sum(correct_vision[indices] & ~correct_text[indices]))
        break_count = int(np.sum(correct_text[indices] & ~correct_vision[indices]))
        result[name] = {"n": int(len(indices)), "fix": fix, "break": break_count,
                        "fix_minus_break": fix - break_count,
                        "fix_minus_break_rate": (fix - break_count) / len(indices) if len(indices) else None,
                        "text_accuracy": float(correct_text[indices].mean()) if len(indices) else None,
                        "vision_accuracy": float(correct_vision[indices].mean()) if len(indices) else None,
                        "confidence_mean": float(confidence[indices].mean()) if len(indices) else None}
    return result


def main(ctx, config):
    ctx.tags.update(stage="phase5", study="A5", diagnostic_revision="v2", device="cpu", post_hoc=True, test_descriptive=True)
    sources = json_read(config["sources"])
    manifest = read_records(sources["manifest"])
    saver = parent(sources, ctx.seed, "saver")
    audit, _ = find_run(sources["phase2_root"], ctx.seed, experiment="phase2_audit", extra_tags={"parent_run_id": saver.name})
    branch = final_epoch(audit / "raw/frozen_test.jsonl")
    text = parent(sources, ctx.seed, "text_only")
    text_rows = final_epoch(text / "raw/test.jsonl")
    rows = align(branch, text_rows)
    pop = population(rows, manifest)
    text_index = index(text_rows)
    confidence = np.array([max(text_index[key]["probabilities"][0]) for key in pop.ids])
    quintile_groups = {f"q{number + 1}": indices for number, indices in enumerate(np.array_split(np.argsort(confidence, kind="stable"), 5))}
    relation_groups = {f"r{label:02d}": np.flatnonzero(pop.gold == label) for label in sorted(set(pop.gold))}
    seed_controls = {}
    for other_seed in (42, 43, 44):
        if other_seed == ctx.seed:
            continue
        other = index(final_epoch(parent(sources, other_seed, "text_only") / "raw/test.jsonl"))
        alternative = np.array([other[key]["prediction"][0] for key in pop.ids])
        seed_controls[f"s{other_seed}"] = {"oracle_f1": oracle(pop, alternative),
                                            "alternative_f1": pop.metrics(alternative)["f1"],
                                            "paired": paired(pop, alternative, pop.text, ctx.seed)}
    maxsim = parent(sources, ctx.seed, "maxsim")
    extracted, _ = find_run(config["phase5_root"], ctx.seed, experiment="phase5_anchor", tag_key="scope", tag_value="full")
    maxsim_rows = final_epoch(extracted / "raw/maxsim_test.jsonl")
    maxsim_pop = population(maxsim_rows, manifest)
    maxsim_flips = {"units": len(maxsim_pop.rows), "prediction_flips": int(np.sum(maxsim_pop.text != maxsim_pop.vision)),
                    "forced_on_vs_own_off": paired(maxsim_pop, maxsim_pop.vision, maxsim_pop.text, ctx.seed),
                    "thresholds": {policy["id"]: {"f1": maxsim_pop.evaluate(policy)["f1"],
                                                  "coverage": maxsim_pop.evaluate(policy)["coverage"],
                                                  "D_post": maxsim_pop.evaluate(policy)["detail"]}
                                   for policy in policies()}}
    values = {"text_f1": pop.metrics(pop.text)["f1"], "forced_on_f1": pop.metrics(pop.vision)["f1"],
              "text_vision_oracle_f1": oracle(pop, pop.vision), "seed_controls": seed_controls,
              "confidence_quintiles": strata(pop, confidence, quintile_groups),
              "relation_strata": strata(pop, confidence, relation_groups), "maxsim_flips": maxsim_flips,
              "oracle_rule": "Choose alternative if correct, else anchor; ties anchor. Uses test labels only for descriptive upper bound, never deployment."}
    ctx.save_json("diagnostics.json", values)
    ctx.set_metrics(values)
