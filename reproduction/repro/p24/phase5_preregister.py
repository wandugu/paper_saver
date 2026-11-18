"""Compute the phase5 dev-only tolerance before any new experiment."""

import argparse
import json
from pathlib import Path

from .phase3_replay import Population, json_read, paired
from .phase4_replay import final_epoch, find_run
from .protocol import digest, read_records


def compute(config):
    result = {"bootstrap_replicates": 2000, "bootstrap_rng": "20261113 + seed",
              "resampling_unit": "post", "confidence": 0.95, "seeds": {},
              "test_used": False, "fallback_constant": 0.02}
    records = read_records(config["manifest"])
    missing = False
    for seed in (42, 43, 44):
        maxsim, _ = find_run(config["phase4_root"], seed, experiment="phase4_extract",
                            tag_key="method", tag_value="maxsim")
        extraction, _ = find_run(config["phase3_root"], seed, experiment="phase3_extract")
        text, _ = find_run(config["phase1_root"], seed, experiment="experiment",
                          tag_key="method", tag_value="text_only", extra_tags={"stage": "formal"})
        maxsim_path = maxsim / "raw/maxsim_dev.jsonl"
        text_path = extraction / "raw/independent_dev.jsonl"
        rows = final_epoch(maxsim_path)
        ids = set(json.loads((text / "data_identity.json").read_text())["split_ids"]["dev"])
        rows = [row for row in rows if row["id"] in ids]
        reference = {row["id"]: row for row in final_epoch(text_path)}
        item = {"maxsim_path": str(maxsim_path), "maxsim_sha256": digest(maxsim_path),
                "text_path": str(text_path), "text_sha256": digest(text_path), "n_units": len(rows)}
        if not rows or any("text_prediction" not in row for row in rows):
            missing = True
            item["all_off_predictions_available"] = False
        else:
            population = Population(rows, records)
            comparison = paired(population, population.text,
                                [reference[key]["prediction"][0] for key in population.ids], seed)
            item.update(all_off_predictions_available=True, comparison=comparison,
                        ci_half_width=(comparison["ci_high"] - comparison["ci_low"]) / 2)
        result["seeds"][str(seed)] = item
    result["delta1"] = 0.02 if missing else max(item["ci_half_width"] for item in result["seeds"].values())
    result["rule"] = "fallback constant: all-off missing" if missing else "maximum paired-bootstrap CI half-width across seeds"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compute(json_read(args.config))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
