"""CPU-only materialization of the missing MaxSim development logits."""

import time
from pathlib import Path

from .phase4_replay import find_run, json_read
from .protocol import cache_key, digest, read_records


def main(ctx, config):
    import torch
    from .encoders import FrozenEncoders
    from .experiment import _forward, verify_resources
    from .model import SaverCore
    from .protocol import FeatureStore

    if torch.cuda.is_available() or config.get("device", "cpu") != "cpu":
        raise RuntimeError("Phase4 MaxSim extraction requires CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(4)
    phase1 = Path(config["phase1_root"])
    parent, parent_row = find_run(phase1, ctx.seed, experiment="experiment", tag_key="method", tag_value="maxsim")
    source = json_read(parent / "config.json")
    source["manifest"] = str(Path(config["source_repo"]) / source["manifest"])
    source["cache"] = str(Path(config["source_repo"]) / source["cache"])
    identity = json_read(parent / "data_identity.json")
    records = read_records(source["manifest"])
    resource_identity = verify_resources(source, "mre")
    dev = sorted([row for row in records if row["split"] == "dev"], key=lambda row: row["id"])
    ctx.tags.update(stage="phase4", study="D2_maxsim_extract", method="maxsim", device="cpu")
    ctx.save_json("input_identity.json", {"parent_run_id": parent_row["run_id"], "checkpoint_sha256": digest(parent / "best.pt"),
                  "manifest_sha256": digest(source["manifest"]), "test_labels_opened": False,
                  "source_repo": config["source_repo"], "resource_identity": resource_identity})
    encoders = FrozenEncoders(source["text_model"], source["vision_model"], device="cpu")

    class ReadOnlyStore(FeatureStore):
        def __init__(self):
            super().__init__(source["cache"], encoders, identity=resource_identity)
            self.hits = self.misses = self.detail_hits = self.detail_misses = 0
            self.memory = {}

        def text_global(self, record, task, visual=True):
            path = self.root / (cache_key(record) + ".pt")
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if not visual or payload.get("global") is not None:
                    self.hits += 1
                    return payload
            self.misses += 1
            units = encoders.encode_text(record, task, return_units=False)
            return {"units": {key: value.detach().cpu() for key, value in units.items() if isinstance(value, torch.Tensor)},
                    "truncated": units["truncated"], "global": encoders.encode_global(record["images"]).cpu() if visual else None}

        def detail(self, image, cache=True):
            import hashlib
            path = self.root / ("detail-" + hashlib.sha256(str(image).encode()).hexdigest() + ".pt")
            if path.exists():
                self.detail_hits += 1
                return torch.load(path, map_location="cpu", weights_only=True)
            if image in self.memory:
                return self.memory[image]
            self.detail_misses += 1
            patches, mask = encoders.encode_detail(image, return_mask=True)
            result = patches[mask.bool()]
            if len(self.memory) < 256:
                self.memory[image] = result
            return result

    store = ReadOnlyStore()
    model = SaverCore(len(identity["labels"]), text_dim=encoders.text_hidden, vision_dim=encoders.vision_hidden)
    model.load_state_dict(torch.load(parent / "best.pt", map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    started = time.monotonic()
    with torch.inference_mode():
        for index, record in enumerate(dev):
            output, _, labels = _forward(model, store, record, "mre", identity["labels"], "maxsim", 1000,
                                         force_details=False, cache_detail=True)
            ctx.write_rows("maxsim_dev", [{"id": record["id"], "gold": labels.tolist(),
                "gate": output["gate"].tolist(), "eligible": output["image_available"].tolist(),
                "text_prediction": output["text_logits"].argmax(-1).tolist(),
                "vision_prediction": output["vision_logits"].argmax(-1).tolist(),
                "prediction": output["logits"].argmax(-1).tolist(), "selected": output["selected"].tolist()}])
            if index % 100 == 0:
                ctx.log_event(kind="frozen_cpu_progress", population="maxsim_dev", completed=index + 1,
                              total=len(dev), elapsed_seconds=time.monotonic() - started)
    ctx.save_json("inference_scope.json", {"device": "cpu", "gpu_hours": 0., "training": False,
                  "read_only_source_cache": str(store.root), "cache_hits": store.hits, "cache_misses": store.misses,
                  "detail_cache_hits": store.detail_hits, "detail_cache_misses": store.detail_misses,
                  "new_feature_cache_bytes": 0, "precision": "float32", "latency_claim": "CPU extraction is not a GPU latency measurement"})
    ctx.set_metrics({"training_updates": 0, "gpu_hours_actual": 0., "source_cache_writes": 0,
                     "dev_posts_complete": True, "dev_rows": len(dev), "seconds": time.monotonic() - started})
