"""Add only missing uncapped frozen features and verify existing files unchanged."""

import shutil
import time
from pathlib import Path

from .encoders import FrozenEncoders
from .experiment import verify_resources
from .phase5_training import AppendOnlyStore, gpu_guard
from .protocol import digest, read_records


def main(ctx, config):
    import torch
    import transformers
    if transformers.__version__ != "4.57.1" or torch.__version__ != "2.6.0+cu118":
        raise RuntimeError("Cache fill requires original pinned runtime")
    ctx.tags.update(stage="phase5", study="cache_fill", device="gpu1", post_hoc=True)
    ctx.save_json("gpu_initial.json", gpu_guard())
    free = shutil.disk_usage("/").free
    if free < 12000000000:
        raise RuntimeError(f"Insufficient free disk for bounded cache fill: {free}")
    identity = verify_resources(config, "mre")
    encoders = FrozenEncoders(config["text_model"], config["vision_model"])
    store = AppendOnlyStore(config["cache"], encoders, identity)
    before = {str(path): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in store.root.glob("*.pt")}
    ctx.save_json("cache_before.json", {"free_bytes": free, "files": before})
    records = read_records(config["manifest"])
    images = set()
    started = time.monotonic()
    for position, record in enumerate(records):
        store.text_global(record, "mre", visual=True)
        for image in record["images"]:
            if image not in images:
                store.detail(image)
                images.add(image)
        if store.new_bytes > 10000000000:
            raise RuntimeError("Cache additions exceeded 10GB")
        if position % 250 == 0:
            ctx.log_event(kind="cache_fill", records=position + 1, total=len(records), images=len(images), new_bytes=store.new_bytes)
    changed = [path for path, value in before.items() if not Path(path).exists() or digest(path) != value["sha256"]]
    if changed:
        raise RuntimeError(f"Existing cache changed: {changed[:3]}")
    ctx.save_json("cache_additions.json", {"new_bytes": store.new_bytes, "files": store.additions,
                  "existing_files_verified": len(before), "changed_existing": changed, "free_bytes_after": shutil.disk_usage("/").free})
    ctx.set_metrics({"new_cache_bytes": store.new_bytes, "new_cache_files": len(store.additions), "existing_files_unchanged": len(before),
                     "seconds": time.monotonic() - started})
