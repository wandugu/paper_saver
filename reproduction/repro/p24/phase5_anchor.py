"""CPU head-only extraction from existing caches, including forced-on MaxSim."""

from pathlib import Path
from types import SimpleNamespace

import torch

from .encoders import FrozenEncoders
from .experiment import verify_resources
from .model import SaverCore
from .phase3_replay import calibration_populations, json_read
from .phase4_replay import find_run
from .phase5_analysis import parent
from .phase5_training import cache_path, evaluate
from .protocol import FeatureStore, cache_key, digest, read_records


class CachedStore(FeatureStore):
    def text_global(self, record, task, visual=True):
        path = cache_path(self.root.parent, self.root / (cache_key(record) + ".pt"))
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if visual and payload.get("global") is None:
            raise RuntimeError(f"Global features absent; cache-fill required: {path}")
        return payload

    def detail(self, image, cache=True):
        import hashlib
        path = cache_path(self.root.parent, self.root / ("detail-" + hashlib.sha256(str(image).encode()).hexdigest() + ".pt"))
        return torch.load(path, map_location=self.encoders.device, weights_only=True)


def main(ctx, config):
    if torch.cuda.is_available():
        raise RuntimeError("Anchor extraction requires CPU")
    torch.set_num_threads(2)
    sources = json_read(config["sources"])
    scope = config["scope"]
    uncapped = scope.startswith("uncapped")
    ctx.tags.update(stage="phase5", scope=scope, study="anchor_extract", device="cpu", post_hoc=True)
    text = parent(sources, ctx.seed, "text_only")
    if uncapped:
        text, _ = find_run(config["phase5_root"], ctx.seed, experiment="phase5_training", tag_key="study", tag_value="A6", extra_tags={"method": "text_only"})
    settings = json_read(text / "config.json")
    for key in ("manifest", "cache", "inventory"):
        settings[key] = str(Path(sources["source_repo"]) / settings[key])
    identity = json_read(text / "data_identity.json")
    manifest = read_records(settings["manifest"])
    dev_ids = set(identity["split_ids"]["dev"])
    dev = [row for row in manifest if row["id"] in dev_ids] if scope == "reduced" else [row for row in manifest if row["split"] == "dev"]
    populations = calibration_populations(manifest, identity, ctx.seed, fractions=(.02, .05, .1) if uncapped else (.02, .05, .1, .2))
    cal_ids = set(identity["split_ids"]["calibration"])
    extension = [row for row in populations[.2] if row["id"] not in cal_ids] if scope == "full" else []
    encoders = SimpleNamespace(device=torch.device("cpu"), span_features=FrozenEncoders.span_features)
    store = CachedStore(settings["cache"], encoders, verify_resources(settings, "mre"))
    for method in (("text_only",) if scope == "uncapped_pilot" else ("text_only", "maxsim")):
        checkpoint = parent(sources, ctx.seed, method) / "best.pt"
        if uncapped:
            directory = text if method == "text_only" else find_run(config["phase5_root"], ctx.seed, experiment="phase5_training", tag_key="study", tag_value="A7_maxsim")[0]
            checkpoint = directory / "best.pt"
        model = SaverCore(len(identity["labels"]))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        for split, records in (("dev", dev), ("calibration_extension", extension)):
            if not records:
                continue
            name = split if method == "text_only" else "maxsim_dev" if split == "dev" else "maxsim_extension"
            rows = evaluate(model, store, records, identity["labels"], method, ctx.seed)
            ctx.write_rows(name, rows)
            ctx.log_event(kind="extracted", method=method, population=name, rows=len(rows))
        if scope in ("full", "uncapped") and method == "maxsim":
            test = [row for row in manifest if row["split"] == "test"]
            ctx.save_json("FORCED_TEST_LOCK.json", {"checkpoint_sha256": digest(checkpoint), "threshold": .5,
                          "force_details": True, "purpose": "Frozen forced-on branches for descriptive replay only"})
            ctx.write_rows("maxsim_test", evaluate(model, store, test, identity["labels"], method, ctx.seed))
        ctx.save_json(f"{method}_identity.json", {"checkpoint": str(checkpoint), "sha256": digest(checkpoint)})
    ctx.set_metrics({"dev_units": len(dev), "extension_units": len(extension), "source_cache_writes": 0,
                     "training_updates": 0, "gpu_hours_actual": 0})
