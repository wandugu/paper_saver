"""Extract reduced-protocol 20% sensitivity branches without cache writes."""

from pathlib import Path
from types import SimpleNamespace

import torch

from .encoders import FrozenEncoders
from .experiment import verify_resources
from .phase3_replay import calibration_populations, json_read
from .phase4_replay import find_run
from .phase5_anchor import CachedStore
from .phase5_training import build_model, evaluate
from .protocol import digest, read_records


def main(ctx, config):
    if torch.cuda.is_available():
        raise RuntimeError("Sensitivity extraction is CPU-only")
    torch.set_num_threads(2)
    study = config["study"]
    ctx.tags.update(stage="phase5", study=study, extraction="calibration_extension", post_hoc=True, device="cpu")
    directory, row = find_run(config["phase5_root"], ctx.seed, experiment="phase5_training", tag_key="study", tag_value=study)
    settings = json_read(directory / "config.json")
    identity = json_read(directory / "data_identity.json")
    records = read_records(settings["manifest"])
    original = set(identity["split_ids"]["calibration"])
    extension = [record for record in calibration_populations(records, identity, ctx.seed)[.2] if record["id"] not in original]
    encoders = SimpleNamespace(device=torch.device("cpu"), span_features=FrozenEncoders.span_features)
    store = CachedStore(settings["cache"], encoders, verify_resources(settings, "mre"))
    model, _ = build_model(settings, ctx.seed)
    model.load_state_dict(torch.load(directory / "best.pt", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    ctx.write_rows("calibration_extension", evaluate(model, store, extension, identity["labels"], "saver", ctx.seed))
    ctx.save_json("input_identity.json", {"parent_run_id": row["run_id"], "checkpoint_sha256": digest(directory / "best.pt"),
                  "extension_ids": [record["id"] for record in extension], "source_cache_writes": 0})
    ctx.set_metrics({"extension_units": len(extension), "training_updates": 0, "source_cache_writes": 0, "gpu_hours_actual": 0})
