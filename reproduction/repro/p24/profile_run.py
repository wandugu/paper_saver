"""Profile a successful, locked P24 checkpoint on the single designated T4."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..common.paths import require_env
from .encoders import FrozenEncoders
from .model import SaverCore
from .profiling import profile
from .protocol import METHODS, digest, read_records

def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON mapping: {path}")
    return value


def _parent_entry(ctx, parent):
    candidates = [Path(ctx.runs_root) / "ledger.jsonl", parent.parent.parent / "ledger.jsonl"]
    entries = []
    for ledger in dict.fromkeys(path.resolve() for path in candidates):
        if ledger.is_file():
            entries.extend(
                row
                for row in read_records(ledger)
                if (ledger.parent / row.get("run_dir", "")).resolve() == parent
            )
    if not entries:
        raise ValueError("Parent training run is not recorded in its ledger")
    entry = entries[-1]
    if (
        entry.get("paper") != "P24"
        or entry.get("experiment") != "experiment"
        or entry.get("status") != "SUCCESS"
        or int(entry.get("seed", -1)) != ctx.seed
    ):
        raise ValueError("Parent must be a SUCCESS P24 training run with the same seed")
    return entry


def _parent_artifacts(ctx, config):
    parent = Path(config["parent_run"]).expanduser().resolve()
    if not (parent / "_SUCCESS").is_file() or any(
        (parent / name).exists() for name in ("FAILED.json", "BLOCKED.json")
    ):
        raise ValueError("Parent training must finish successfully before profiling")
    entry = _parent_entry(ctx, parent)
    parent_config = _json(parent / "config.json")
    task, method = parent_config["task"], parent_config["method"]
    if task not in {"mre", "mner"} or method not in METHODS or parent_config.get("pilot"):
        raise ValueError("Parent must be a formal registered P24 task/method")
    for key, expected in (("task", task), ("method", method)):
        if key in config and config[key] != expected:
            raise ValueError(f"Profile {key} must match the parent; overrides are forbidden")
    checkpoint = parent / "best.pt"
    checkpoint_hash = digest(checkpoint)
    config_hash = digest(parent / "config.json")
    for key, actual in (
        ("checkpoint_sha256", checkpoint_hash),
        ("parent_config_sha256", config_hash),
    ):
        if config.get(key, actual) != actual:
            raise ValueError(f"Requested {key} does not match the parent artifact")
    lock = _json(parent / "METHOD_LOCK.json")
    calibration = _json(parent / "calibration.json")
    if (
        lock.get("seed") != ctx.seed
        or lock.get("method") != method
        or lock.get("best_checkpoint_sha256") != checkpoint_hash
        or lock.get("threshold") != calibration.get("threshold")
        or "threshold" not in calibration
    ):
        raise ValueError("Checkpoint/calibration differ from the frozen parent method lock")
    threshold = calibration["threshold"]
    if threshold is not None and (
        isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid frozen calibration threshold")
    matched = _json(parent / "matched_random.json")
    rate = matched["rate"]
    if isinstance(rate, bool) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Invalid frozen matched-random activation rate")
    if method == "random" and (
        not matched.get("saver_run")
        or calibration.get("matching_run") != matched["saver_run"]
        or calibration.get("matching_probability") != rate
    ):
        raise ValueError("Random profile requires its original SAVER matching run/rate")
    identity = _json(parent / "data_identity.json")
    manifest = Path(parent_config["manifest"]).expanduser().resolve()
    if digest(manifest) != identity["sha256"]:
        raise ValueError("Manifest differs from the training data identity")
    mapping = identity["labels"]
    if (
        not isinstance(mapping, dict)
        or mapping.get("None") != 0
        or sorted(mapping.values()) != list(range(len(mapping)))
    ):
        raise ValueError("Parent label mapping must be contiguous with None=0")
    dev_ids = identity["split_ids"]["dev"]
    if not dev_ids or len({str(value) for value in dev_ids}) != len(dev_ids):
        raise ValueError("Parent dev IDs must be nonempty and unique")
    selected = sorted(dev_ids, key=str)[:32]
    wanted = {str(value) for value in selected}
    found = {}
    for record in read_records(manifest):
        key = str(record["id"])
        if key not in wanted:
            continue
        if key in found or record["split"] not in {"dev", "val", "valid"}:
            raise ValueError("A selected parent dev ID is duplicated or not an official dev record")
        if record.get("task", task) != task:
            raise ValueError("Selected dev record has the wrong task")
        found[key] = record
    if set(found) != wanted:
        raise ValueError("Some parent dev IDs are missing from the bound manifest")
    records = [found[str(value)] for value in selected]
    provenance = {
        "parent_run": str(parent),
        "parent_run_id": entry["run_id"],
        "seed": ctx.seed,
        "task": task,
        "method": method,
        "parent_config_sha256": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": identity["sha256"],
        "calibration_sha256": digest(parent / "calibration.json"),
        "data_identity_sha256": digest(parent / "data_identity.json"),
        "method_lock_sha256": digest(parent / "METHOD_LOCK.json"),
        "matched_random_sha256": digest(parent / "matched_random.json"),
        "threshold": threshold,
        "random_rate": rate,
        "matching_saver_run": matched.get("saver_run"),
        "dev_ids": selected,
        "sample_rule": "First 32 parent dev IDs sorted by str(id)",
        "test_evaluations": 0,
        "calibration_refits": 0,
    }
    return parent_config, mapping, records, provenance, checkpoint


def _gpu_guard():
    if socket.gethostname().split(".")[0] != require_env("P24_FORMAL_HOSTNAME"):
        raise RuntimeError("P24 profiling is restricted to the configured formal hostname")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P24 profiling requires CUDA_VISIBLE_DEVICES=0, not another GPU")
    gpu_uuid = require_env("P24_PROFILE_GPU_UUID")
    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=index,name,uuid,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError("Cannot verify the designated profiling GPU")
    index, name, uuid, used = [value.strip() for value in rows[0]]
    if index != "0" or uuid != gpu_uuid or "T4" not in name:
        raise RuntimeError("Physical GPU identity differs from the configured T4 GPU0")
    if int(used) >= 500:
        raise RuntimeError(
            "Profiling GPU0 is not idle (<500 MiB required); leave all processes untouched"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device is required for profiling")
    probe = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize(0)
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    process_rows = list(csv.reader(processes.stdout.strip().splitlines(), skipinitialspace=True))
    if any(
        len(row) == 2 and row[1].strip() == gpu_uuid and row[0].strip() != str(os.getpid())
        for row in process_rows
    ):
        raise RuntimeError("Another compute process occupies profiling GPU0; leave it untouched")
    own_devices = {
        row[1].strip()
        for row in process_rows
        if len(row) == 2 and row[0].strip() == str(os.getpid())
    }
    if own_devices != {gpu_uuid}:
        raise RuntimeError("CUDA ordinal 0 does not map to the designated physical GPU UUID")
    del probe
    return {
        "ssh_alias": os.environ.get("P24_FORMAL_HOST", "formal-host"),
        "hostname": socket.gethostname(),
        "physical_gpu": 0,
        "gpu_uuid": uuid,
        "gpu_name": name,
        "cuda_visible_devices": "0",
        "initial_memory_mib": int(used),
        "torch_version": torch.__version__,
    }


def _state_and_model_config(checkpoint, parent_config, mapping):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Parent checkpoint must contain a state mapping")
    model_config = dict(parent_config.get("model_kwargs", {}))
    if "state_dict" in payload or "model_state_dict" in payload:
        declared_mapping = payload.get("mapping", mapping)
        if declared_mapping != mapping:
            raise ValueError("Checkpoint label mapping differs from parent data identity")
        checkpoint_config = payload.get("cfg", payload.get("config", {}))
        for key in ("task", "method"):
            if key in checkpoint_config and checkpoint_config[key] != parent_config[key]:
                raise ValueError("Checkpoint configuration differs from the parent")
        model_config.update(checkpoint_config.get("model_kwargs", {}))
        state = payload.get("state_dict", payload.get("model_state_dict"))
    else:
        state = payload
    for key in ("task", "text_dim", "vision_dim", "num_labels"):
        model_config.pop(key, None)
    return state, model_config


def main(ctx, config):
    parent_config, mapping, records, provenance, checkpoint = _parent_artifacts(ctx, config)
    task, method = provenance["task"], provenance["method"]
    ctx.tags.update(
        task=task, method=method, stage="profile", parent_run_id=provenance["parent_run_id"]
    )
    ctx.save_json("parent_identity.json", provenance)
    lock_path = Path(ctx.runs_root) / ".p24-profile-gpu0.lock"
    import fcntl

    with lock_path.open("a") as profile_lock:
        try:
            fcntl.flock(profile_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another P24 GPU0 profiler holds the lock; serialize profile jobs"
            ) from exc
        hardware = _gpu_guard()
        ctx.save_json("profile_hardware.json", hardware)
        random.seed(ctx.seed)
        np.random.seed(ctx.seed)
        torch.manual_seed(ctx.seed)
        torch.cuda.manual_seed_all(ctx.seed)
        torch.set_num_threads(4)
        state, model_config = _state_and_model_config(checkpoint, parent_config, mapping)
        encoders = FrozenEncoders(
            parent_config["text_model"], parent_config["vision_model"], "cuda:0"
        )
        model = SaverCore(
            len(mapping),
            task=task,
            text_dim=encoders.text_hidden,
            vision_dim=encoders.vision_hidden,
            **model_config,
        ).to("cuda:0")
        model.load_state_dict(state, strict=True)
        model.eval()
        threshold = math.inf if provenance["threshold"] is None else provenance["threshold"]
        metrics = profile(
            ctx,
            model,
            encoders,
            None,
            records,
            task,
            mapping,
            method,
            threshold,
            provenance["random_rate"],
        )
        ctx.set_metrics(
            {
                **metrics,
                "parent_run_id": provenance["parent_run_id"],
                "parent_config_sha256": provenance["parent_config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "profile_gpu_uuid": hardware["gpu_uuid"],
                "test_evaluations": 0,
            }
        )
