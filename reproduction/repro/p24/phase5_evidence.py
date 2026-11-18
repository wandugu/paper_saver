"""Export lossless decision-relevant projections without images or checkpoints."""

import gzip
import json
from pathlib import Path

import torch

from .phase3_replay import calibration_populations, json_read
from .phase5_analysis import parent
from .protocol import digest, post_key, read_records


def project(row):
    fields = ("id", "epoch", "gold", "prediction", "text_prediction", "vision_prediction", "gate", "eligible", "selected", "utility_target", "text_ce", "vision_ce")
    result = {key: row[key] for key in fields if key in row}
    probabilities = row.get("text_probabilities", row.get("probabilities"))
    if probabilities is not None:
        result["max_text_probability"] = [max(values) for values in probabilities]
    return result


def gzip_rows(path, rows):
    with path.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as compressed:
            for row in rows:
                compressed.write((json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode())


def main(ctx, config):
    if torch.cuda.is_available():
        raise RuntimeError("Evidence export is CPU only")
    ctx.tags.update(stage="phase5", study="prediction_evidence", device="cpu", post_hoc=True)
    root = Path(config["phase5_root"])
    sources = json_read(config["sources"])
    manifest = read_records(sources["manifest"])
    ledger = [row for row in read_records(root / "ledger.jsonl") if row["status"] == "SUCCESS"]
    paths = set()
    population_index = {}
    for row in ledger:
        directory = root / row["run_dir"]
        if row["experiment"] in ("phase5_training", "phase5_anchor", "phase5_extension"):
            paths.update(path for path in (directory / "raw").glob("*.jsonl") if "epoch" not in path.name and path.name != "training.jsonl")
        if row["experiment"] == "phase5_analysis":
            identity = json_read(directory / "source_identity.json")
            trained = Path(identity["trained_run"])
            split_identity = json_read(trained / "data_identity.json")
            fractions = (.02, .05, .1) if row["tags"]["study"] == "A7" else (.02, .05, .1, .2)
            populations = calibration_populations(manifest, split_identity, row["seed"], fractions=fractions)
            population_index[row["run_id"]] = {f"f{round(fraction * 100):02d}": [record["id"] for record in records] for fraction, records in populations.items()}
            paths.update((trained / "raw").glob("*.jsonl"))
            paths.update((Path(identity["anchor_run"]) / "raw").glob("*.jsonl"))
    delta = json_read("repro/p24/phase5_delta1.json")
    for values in delta["seeds"].values():
        paths.update((Path(values["maxsim_path"]), Path(values["text_path"])))
    for seed in (42, 43, 44):
        for method in ("text_only", "maxsim"):
            directory = parent(sources, seed, method)
            paths.add(directory / "raw/calibration.jsonl")
            if method == "text_only":
                paths.add(directory / "raw/test.jsonl")
    paths = {path for path in paths if path.is_file() and "epoch" not in path.name and path.name != "training.jsonl"}
    target = ctx.run_dir / "projections"
    target.mkdir()
    entries, written = [], {}
    for path in sorted(paths):
        raw_hash = digest(path)
        if raw_hash not in written:
            rows = read_records(path)
            if not rows or "id" not in rows[0]:
                continue
            destination = target / f"{raw_hash}.jsonl.gz"
            gzip_rows(destination, (project(row) for row in rows))
            written[raw_hash] = {"projection": str(destination.relative_to(ctx.run_dir)), "rows": len(rows),
                                 "bytes": destination.stat().st_size, "sha256": digest(destination)}
        entries.append({"source": str(path), "source_sha256": raw_hash, **written[raw_hash]})
    manifest_path = target / "manifest_posts.jsonl.gz"
    gzip_rows(manifest_path, ({"id": row["id"], "post_key": post_key(row), "images": row["images"], "split": row["split"]} for row in manifest))
    ctx.save_json("calibration_population_ids.json", population_index)
    ctx.save_json("prediction_evidence_index.json", {"sources": entries, "manifest_source": sources["manifest"],
                  "manifest_source_sha256": digest(sources["manifest"]), "manifest_projection_sha256": digest(manifest_path),
                  "projection_rule": "Keep exact labels, predictions, gate, eligibility, selected image indices and CE targets/losses. Replace full probability vectors only by their exact maxima, which is sufficient for the fixed 1-max-p scorer. No logits/images/checkpoints. Preserve epoch tags and original source hashes.",
                  "formal_runtime_only": True, "excluded_snapshot_policy": "Old phase5 roots are not read; historical pre-phase5 source anchors remain included."})
    size = sum(path.stat().st_size for path in target.iterdir())
    if size > 6000000:
        raise RuntimeError(f"Compact predictions exceed reserved 6MB evidence allocation: {size}")
    ctx.set_metrics({"source_files": len(entries), "unique_prediction_files": len(written), "projection_bytes": size,
                     "projected_rows": sum(value["rows"] for value in written.values()), "training_updates": 0})
