"""Collection of lightweight, failure-tolerant run environment metadata."""

from __future__ import annotations

import getpass
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd=None):
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception:
        return None


def _distribution_version(*names):
    for name in names:
        try:
            return importlib.metadata.version(name)
        except Exception:
            continue
    return None


def _nvidia_info(cuda_visible):
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    result = _run(
        [
            exe,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if result is None or result.returncode != 0:
        return None
    visible = None
    if cuda_visible:
        try:
            visible = {int(x.strip()) for x in cuda_visible.split(",") if x.strip() and x.strip() != "-1"}
        except Exception:
            visible = None
    records = []
    for line in result.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            index = int(fields[0])
        except Exception:
            index = fields[0]
        if visible is not None and index not in visible:
            continue
        records.append(
            {
                "index": index,
                "name": fields[1],
                "memory.total": fields[2],
                "driver_version": fields[3],
            }
        )
    return records


def collect_env(repo_root) -> dict:
    """Collect environment metadata; no individual probe failure raises."""

    root = Path(repo_root)
    commit = None
    dirty_files = []
    status = _run(["git", "rev-parse", "HEAD"], cwd=root)
    if status is not None and status.returncode == 0:
        commit = status.stdout.strip() or None
    status = _run(["git", "status", "--porcelain"], cwd=root)
    if status is not None and status.returncode == 0:
        dirty_files = [line[3:] if len(line) >= 3 else line for line in status.stdout.splitlines() if line]
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    versions = {
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "numpy": _distribution_version("numpy"),
    }
    torch_cuda_available = None
    torch_device_names = []
    if versions["torch"] is not None:
        try:
            import torch  # type: ignore

            torch_cuda_available = bool(torch.cuda.is_available())
            if torch_cuda_available:
                for i in range(torch.cuda.device_count()):
                    try:
                        torch_device_names.append(torch.cuda.get_device_name(i))
                    except Exception:
                        torch_device_names.append(None)
        except Exception:
            torch_cuda_available = None
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_files),
        "git_changed_files": dirty_files,
        "hostname": platform.node(),
        "user": getpass.getuser(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "cuda_visible_devices": cuda_visible,
        "nvidia_smi": _nvidia_info(cuda_visible),
        "versions": versions,
        "torch_cuda_available": torch_cuda_available,
        "torch_device_names": torch_device_names,
    }


__all__ = ["collect_env"]

