"""Phase5 training with hard dev selection and auditable frozen branches."""

import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re

import numpy as np
import torch

from ..common.paths import require_env
from .encoders import FrozenEncoders
from .experiment import _forward, _mapping, verify_resources
from .model import SaverCore
from .protocol import FeatureStore, cache_key, digest, read_records, split_records, task_f1


FROZEN_MODULES = ("width_embedding", "span_projection", "pair_projection", "classifier")


@lru_cache(maxsize=1)
def excluded_cache_entries():
    """Read and validate cache identities relative to the effective feature root."""
    path = Path(__file__).with_name("phase5_runtime_exclusions.json")
    entries = json.loads(path.read_text(encoding="utf-8"))["invalid_cache_paths"]
    pattern = re.compile(r"[0-9a-f]{24}/(?:detail-)?[0-9a-f]{64}\.pt")
    if not entries or any(not isinstance(entry, str) or not pattern.fullmatch(entry) for entry in entries):
        raise ValueError("Invalid phase 5 exclusions: expected namespace/file paths relative to the feature root")
    return frozenset(PurePosixPath(entry) for entry in entries)


def cache_path(feature_root, path):
    feature_root, path = Path(feature_root), Path(path)
    try:
        relative = path.relative_to(feature_root)
    except ValueError as error:
        raise ValueError(f"Feature cache path is outside its configured root: {path}") from error
    if len(relative.parts) != 2 or ".." in relative.parts:
        raise ValueError(f"Expected a namespace and file within the feature cache: {path}")
    if PurePosixPath(relative.as_posix()) in excluded_cache_entries():
        return path.with_name("phase5-tf4571-" + path.name)
    return path


def build_model(config, seed, text_dim=768, vision_dim=768):
    variant = config.get("variant", "control")
    model = SaverCore(config["num_labels"], text_dim=text_dim, vision_dim=vision_dim,
                      text_ce_weight=1.0 if variant == "A1" else 0.0,
                      route_dropout=0.3 if variant in ("A1", "A2") else 0.0,
                      anchor_residual=variant == "A2")
    loading = None
    if variant == "A2":
        checkpoint = Path(config["anchor_checkpoint"])
        loading = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
        expected = {"visual_residual_head.weight", "visual_residual_head.bias"}
        if set(loading.missing_keys) != expected or loading.unexpected_keys:
            raise ValueError(f"Unexpected anchor checkpoint keys: {loading}")
        for name, parameter in model.named_parameters():
            if name.split(".")[0] in FROZEN_MODULES:
                parameter.requires_grad_(False)
        loading = {"missing": loading.missing_keys, "unexpected": loading.unexpected_keys,
                   "anchor_checkpoint": str(checkpoint), "anchor_sha256": digest(checkpoint),
                   "frozen_parameters": [name for name, parameter in model.named_parameters() if not parameter.requires_grad]}
    return model, loading


class AppendOnlyStore(FeatureStore):
    def __init__(self, root, encoders, identity):
        super().__init__(root, encoders, identity)
        self.new_bytes = 0
        self.additions = []

    def create(self, path, value):
        if self.new_bytes > 10000000000:
            raise RuntimeError("Cache addition limit exceeded")
        try:
            with path.open("xb") as stream:
                torch.save(value, stream)
            size = path.stat().st_size
            self.new_bytes += size
            self.additions.append({"path": str(path), "bytes": size, "sha256": digest(path)})
        except FileExistsError:
            pass

    def text_global(self, record, task, visual=True):
        path = cache_path(self.root.parent, self.root / (cache_key(record) + ".pt"))
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if visual and payload.get("global") is None:
                payload = dict(payload)
                payload["global"] = self.encoders.encode_global(record["images"]).cpu()
            return payload
        units = self.encoders.encode_text(record, task, return_units=False)
        payload = {"units": {name: value.detach().cpu() for name, value in units.items() if isinstance(value, torch.Tensor)},
                   "truncated": units["truncated"],
                   "global": self.encoders.encode_global(record["images"]).cpu() if visual else None}
        self.create(path, payload)
        return payload

    def detail(self, image, cache=True):
        path = cache_path(self.root.parent, self.root / ("detail-" + hashlib.sha256(str(image).encode()).hexdigest() + ".pt"))
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True).to(self.encoders.device)
        patches, mask = self.encoders.encode_detail(image, return_mask=True)
        result = patches[mask.bool()]
        if cache:
            self.create(path, result.detach().cpu())
        return result


def evaluate(model, store, records, mapping, method, seed, *, retain_logits=False):
    model.eval()
    rows = []
    with torch.inference_mode():
        for record in records:
            output, _, labels = _forward(model, store, record, "mre", mapping, method, 1000,
                                         threshold=.5, seed=seed, force_details=method != "text_only")
            text_probability = output["text_logits"].softmax(-1)
            row = {"id": record["id"], "gold": labels.tolist(),
                   "prediction": output["logits"].argmax(-1).tolist(),
                   "text_prediction": output["text_logits"].argmax(-1).tolist(),
                   "vision_prediction": output["vision_logits"].argmax(-1).tolist(),
                   "text_probabilities": text_probability.tolist(),
                   "gate": output["gate"].tolist(), "eligible": output["image_available"].tolist(),
                   "selected": [[0] if valid else [] for valid in output["image_available"].tolist()],
                   "utility_target": output["utility_targets"].tolist(),
                   "text_ce": output["text_losses"].tolist(), "vision_ce": output["vision_losses"].tolist()}
            if retain_logits:
                row.update(text_logits=output["text_logits"].tolist(), vision_logits=output["vision_logits"].tolist(),
                           logits=output["logits"].tolist())
            rows.append(row)
    return rows


def all_off_f1(rows):
    return task_f1([{**row, "prediction": row["text_prediction"]} for row in rows], "mre")


def gpu_guard():
    if socket.gethostname().split(".")[0] != require_env("P24_FORMAL_HOSTNAME") or os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Phase5 GPU jobs require the configured formal hostname and physical GPU1")
    output = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], text=True)
    fields = [field.strip() for field in output.strip().split(",")]
    if int(fields[1]) >= 1536 or int(fields[2]) > 80:
        raise RuntimeError(f"GPU1 not eligible: {fields}")
    return {"physical_gpu": 1, "uuid": fields[0], "initial_memory_mib": int(fields[1]),
            "temperature_c": int(fields[2]), "pid": os.getpid(), "timestamp": time.time()}


def main(ctx, config):
    import transformers
    if transformers.__version__ != "4.57.1" or torch.__version__ != "2.6.0+cu118":
        raise RuntimeError("Formal phase5 requires original pinned torch/transformers runtime")
    started = time.monotonic()
    ctx.save_json("gpu_initial.json", gpu_guard())
    ctx.tags.update(stage="phase5", study=config["study"], method=config["method"], variant=config.get("variant", "control"), post_hoc=True)
    torch.set_num_threads(4)
    random.seed(ctx.seed)
    np.random.seed(ctx.seed)
    torch.manual_seed(ctx.seed)
    torch.cuda.manual_seed_all(ctx.seed)
    records = read_records(config["manifest"])
    splits = split_records(records, ctx.seed, fit_cap=config.get("fit_cap", 4096), dev_cap=config.get("dev_cap", 512))
    mapping = _mapping([record for record in records if record["split"] == "train"], "mre")
    if len(mapping) != config["num_labels"]:
        raise ValueError("Label mapping mismatch")
    identity = verify_resources(config, "mre")
    ctx.save_json("data_identity.json", {"sha256": digest(config["manifest"]), "labels": mapping,
                  "split_ids": {name: [record["id"] for record in rows] for name, rows in splits.items()}})
    ctx.save_json("disk_before.json", {"free_bytes": __import__("shutil").disk_usage("/").free})
    encoders = FrozenEncoders(config["text_model"], config["vision_model"])
    store = AppendOnlyStore(config["cache"], encoders, identity)
    model, loading = build_model(config, ctx.seed, encoders.text_hidden, encoders.vision_hidden)
    model.cuda()
    anchor_state = {name: value.clone() for name, value in model.state_dict().items() if name.split(".")[0] in FROZEN_MODULES} if loading else None
    if loading:
        ctx.save_json("anchor_loading.json", loading)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=1e-4, betas=(.9, .999), weight_decay=.01)
    maximum_steps = math.ceil(len(splits["fit"]) / 16) * 10
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / max(1, .1 * maximum_steps), 1) *
        (.5 + .5 * math.cos(math.pi * max(0, step - .1 * maximum_steps) / (.9 * maximum_steps))))
    best, stale, step = -1., 0, 0
    best_info = None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(10):
        model.train()
        epoch_records = list(splits["fit"])
        random.Random(ctx.seed + epoch).shuffle(epoch_records)
        optimizer.zero_grad(set_to_none=True)
        for position, record in enumerate(epoch_records):
            output, _, _ = _forward(model, store, record, "mre", mapping, config["method"], step, training=True, seed=ctx.seed)
            if not torch.isfinite(output["loss"]):
                raise RuntimeError("Nonfinite training loss; no tuning fallback")
            (output["loss"] / 16).backward()
            if (position + 1) % 16 == 0 or position + 1 == len(epoch_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    progress = {"epoch": epoch, "step": step, "loss": float(output["loss"]),
                                "text_loss": float(output["text_loss"]), "utility_loss": float(output["utility_loss"]),
                                "gate_mean": float(output["gate"].mean()), "eta_mean": float(output["eta"].mean()),
                                "elapsed_seconds": time.monotonic() - started}
                    ctx.write_rows("training", [progress])
                    print(json.dumps(progress), flush=True)
        rows = evaluate(model, store, splits["dev"], mapping, config["method"], ctx.seed, retain_logits=True)
        score = task_f1(rows, "mre")
        info = {"epoch": epoch, "step": step, "dev_f1": score, "all_off_dev_f1": all_off_f1(rows),
                "gate_mean": float(np.mean([row["gate"][0] for row in rows]))}
        ctx.log_event(kind="development", **info)
        ctx.write_rows(f"dev_epoch_{epoch:02d}", rows)
        print(json.dumps(info), flush=True)
        if score > best:
            best, stale = score, 0
            checkpoint = ctx.run_dir / f"checkpoint_epoch_{epoch:02d}.pt"
            torch.save(model.state_dict(), checkpoint)
            best_info = {**info, "checkpoint": str(checkpoint)}
        else:
            stale += 1
        if step >= 1000 and stale >= 3:
            break
    model.load_state_dict(torch.load(best_info["checkpoint"], weights_only=True))
    if anchor_state is not None:
        if any(not torch.equal(model.state_dict()[name], value) for name, value in anchor_state.items()):
            raise RuntimeError("Frozen anchor changed")
    torch.save(model.state_dict(), ctx.run_dir / "best.pt")
    ctx.save_json("best.json", {**best_info, "checkpoint_sha256": digest(ctx.run_dir / "best.pt"), "anchor_unchanged": anchor_state is not None})
    dev_rows = evaluate(model, store, splits["dev"], mapping, config["method"], ctx.seed, retain_logits=True)
    ctx.write_rows("dev", dev_rows)
    calibration = evaluate(model, store, splits["calibration"], mapping, config["method"], ctx.seed)
    ctx.write_rows("calibration", calibration)
    ctx.save_json("METHOD_LOCK.json", {"checkpoint_sha256": digest(ctx.run_dir / "best.pt"), "threshold": .5,
                  "test_role": "descriptive post hoc; no configuration or seed selection"})
    test_rows = evaluate(model, store, splits["test"], mapping, config["method"], ctx.seed)
    ctx.write_rows("test", test_rows)
    ctx.save_json("cache_additions.json", {"new_bytes": store.new_bytes, "files": store.additions})
    ctx.set_metrics({"development_f1": best, "all_off_dev_f1": best_info["all_off_dev_f1"],
                     "test_f1": task_f1(test_rows, "mre"), "optimizer_steps": step,
                     "fit_units": len(splits["fit"]), "dev_units": len(splits["dev"]),
                     "calibration_units": len(splits["calibration"]), "test_units": len(splits["test"]),
                     "seconds": time.monotonic() - started, "new_cache_bytes": store.new_bytes,
                     "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2})
