"""Run ledger and context management for reproducible experiments."""
from __future__ import annotations

import copy
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(dt: Optional[datetime] = None) -> str:
    dt = dt or _utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    # os.replace is atomic and works on Windows when destination does not remain open.
    os.replace(tmp, path)


def _deep_merge(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge src into dst recursively, mutating and returning dst."""
    for key, val in src.items():
        if isinstance(val, Mapping) and isinstance(dst.get(key), Mapping):
            child = dict(dst[key])
            _deep_merge(child, val)
            dst[key] = child
        elif isinstance(val, Mapping):
            dst[key] = _deep_merge({}, val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        # Linux: lock ledger writes to avoid interleaved records. On Windows no lock is used.
        if os.name != "nt":
            try:
                import fcntl  # type: ignore
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                fcntl = None  # type: ignore
            try:
                fh.write(line)
                fh.flush()
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
        else:
            fh.write(line)


def is_superseded(run_dir: Union[str, Path]) -> bool:
    """True when a finished run was retired by hand with ``SUPERSEDED.json``.

    The ledger is append-only, so a run later found invalid (for example a
    SUCCESS produced by a since-fixed runner bug) keeps its ledger row; the
    marker file in its run directory, which records the reason, keeps it out
    of tables and dependency lookups.
    """
    return (Path(run_dir) / "SUPERSEDED.json").exists()


def _resolve_repo_root() -> Path:
    # ledger.py lives in repro/common; its grandparent is the repository root.
    return Path(__file__).resolve().parents[2]


def _visible_gpu_count() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or value.strip() == "":
        # An unset mask means every CUDA device visible to the process. Probe
        # torch only at resource-recording time; the framework remains usable
        # when torch is absent.
        try:
            import torch  # type: ignore

            cuda = getattr(torch, "cuda", None)
            if cuda is not None and bool(cuda.is_available()):
                return max(0, int(cuda.device_count()))
        except Exception:
            pass
        return 0
    text = value.strip()
    if text in {"-1", "none", "None"}:
        return 0
    return len([part for part in text.split(",") if part.strip() and part.strip() not in {"-1"}])


def _torch_resources() -> Dict[str, Any]:
    """Collect optional torch CUDA peak allocation without requiring torch."""
    result: Dict[str, Any] = {"torch_cuda_max_memory_allocated": {}, "gpu_hours": 0.0}
    try:
        import torch  # type: ignore
    except Exception:
        result["gpu_hours"] = 0.0
        return result
    try:
        cuda = getattr(torch, "cuda", None)
        available = bool(cuda is not None and cuda.is_available())
    except Exception:
        available = False
    if not available:
        result["gpu_hours"] = 0.0
        return result
    count = 0
    try:
        count = int(cuda.device_count())
    except Exception:
        count = _visible_gpu_count()
    # Peak memory is reported per visible CUDA device. Values are bytes as returned by torch.
    peaks: Dict[str, Any] = {}
    for index in range(max(0, count)):
        try:
            peaks[str(index)] = int(cuda.max_memory_allocated(index))
        except Exception:
            continue
    result["torch_cuda_max_memory_allocated"] = peaks
    result["gpu_hours"] = float(_visible_gpu_count() * 0.0)  # replaced by caller with wall hours
    return result


class RunContext:
    """Context manager recording a reproducible experiment run.

    All metric updates must go through :meth:`set_metrics`; final metrics and
    resource accounting are written automatically on context exit.
    """

    def __init__(
        self,
        paper: str,
        experiment: str,
        config: Mapping[str, Any],
        seed: int,
        runs_root: Optional[Union[str, os.PathLike[str]]] = None,
        prereg: Optional[Union[str, os.PathLike[str], Mapping[str, Any]]] = None,
        tags: Optional[Mapping[str, Any]] = None,
        prereg_sha256: Optional[str] = None,
    ) -> None:
        self.paper = str(paper)
        self.experiment = str(experiment)
        self.config = copy.deepcopy(dict(config))
        self.seed = int(seed)
        if runs_root is None:
            runs_root = os.environ.get("REPRO_RUNS_ROOT")
        self.runs_root = Path(runs_root) if runs_root else _resolve_repo_root() / "tmp" / "repro_runs"
        self.runs_root = self.runs_root.expanduser()
        self.tags = copy.deepcopy(dict(tags or {}))
        self.prereg_source = prereg
        self.prereg: Optional[Dict[str, Any]] = None
        self.prereg_sha256 = prereg_sha256
        self.run_id = ""
        self.run_dir = Path()
        self._metrics: Dict[str, Any] = {}
        self._started_dt: Optional[datetime] = None
        self._finished_dt: Optional[datetime] = None
        self._entered = False
        self._status: Optional[str] = None
        self._env: Dict[str, Any] = {}

    @property
    def metrics(self) -> Dict[str, Any]:
        """Current metrics (deep copy to avoid accidental out-of-band mutation)."""
        return copy.deepcopy(self._metrics)

    @property
    def status(self) -> Optional[str]:
        return self._status

    def __enter__(self) -> "RunContext":
        if self._entered:
            return self
        self._entered = True
        self._started_dt = _utc_now()
        timestamp = self._started_dt.strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{self.paper}_{self.experiment}_s{self.seed}_{timestamp}_{secrets.token_hex(3)}"
        self.run_dir = self.runs_root / self.paper / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        if self.prereg_source is not None:
            try:
                if isinstance(self.prereg_source, Mapping):
                    self.prereg = copy.deepcopy(dict(self.prereg_source))
                else:
                    from .prereg import load_prereg_with_sha256
                    self.prereg, actual_sha256 = load_prereg_with_sha256(self.prereg_source)
                    if self.prereg_sha256 is not None and actual_sha256 != self.prereg_sha256:
                        raise ValueError("Preregistration SHA-256 mismatch")
                    self.prereg_sha256 = actual_sha256
            except Exception:
                # Invalid preregistration should be surfaced to caller before run body.
                self._entered = False
                raise
        try:
            from .envinfo import collect_env
            self._env = collect_env(_resolve_repo_root())
        except Exception:
            self._env = {}
        self._env.setdefault("git_dirty", bool(self._env.get("git_dirty", False)))
        _json_dump(self.run_dir / "config.json", self.config)
        _json_dump(self.run_dir / "env.json", self._env)
        if self.prereg is not None and self.prereg_sha256 is not None:
            _json_dump(self.run_dir / "preregistration.json", {"id": self.prereg["id"], "sha256": self.prereg_sha256})
        (self.run_dir / "started_at").write_text(_stamp(self._started_dt) + "\n", encoding="utf-8")
        return self

    def _raw_path(self, name: str) -> Path:
        # Keep caller supplied names within raw/ and prevent path traversal.
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("raw name must be a relative path")
        path = self.run_dir / "raw" / relative
        return path if path.name.endswith(".jsonl") else Path(str(path) + ".jsonl")

    def write_rows(self, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append dictionaries as JSONL under ``raw/{name}.jsonl``."""
        if not self._entered:
            raise RuntimeError("RunContext must be entered before writing rows")
        path = self._raw_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError("rows must contain mappings")
                fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str) + "\n")

    def log_event(self, **kw: Any) -> None:
        if not self._entered:
            raise RuntimeError("RunContext must be entered before logging events")
        event = dict(kw)
        event["time"] = _stamp()
        _append_jsonl(self.run_dir / "events.jsonl", event)

    def save_text(self, rel: Union[str, os.PathLike[str]], text: str) -> None:
        if not self._entered:
            raise RuntimeError("RunContext must be entered before saving files")
        path = self._safe_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text), encoding="utf-8")

    def save_json(self, rel: Union[str, os.PathLike[str]], obj: Any) -> None:
        if not self._entered:
            raise RuntimeError("RunContext must be entered before saving files")
        _json_dump(self._safe_path(rel), obj)

    def _safe_path(self, rel: Union[str, os.PathLike[str]]) -> Path:
        path = Path(rel)
        if path.name == "metrics.json":
            raise ValueError("metrics.json is managed by set_metrics")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be relative to run directory")
        return self.run_dir / path

    def set_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        _deep_merge(self._metrics, metrics)

    def _resource_metrics(self, wall_seconds: float) -> Dict[str, Any]:
        wall_hours = max(0.0, float(wall_seconds) / 3600.0)
        resources: Dict[str, Any] = {
            "wall_seconds": float(wall_seconds),
            "wall_hours": wall_hours,
            "gpu_hours": wall_hours * _visible_gpu_count(),
        }
        try:
            import torch  # type: ignore
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and bool(cuda.is_available()):
                peaks: Dict[str, Any] = {}
                try:
                    count = int(cuda.device_count())
                except Exception:
                    count = 0
                if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                    resources["gpu_hours"] = wall_hours * max(0, count)
                for index in range(max(0, count)):
                    try:
                        peaks[str(index)] = int(cuda.max_memory_allocated(index))
                    except Exception:
                        continue
                resources["torch_cuda_max_memory_allocated"] = peaks
        except Exception:
            pass
        return resources

    def _write_index(self, status: str, gates: Optional[Mapping[str, Any]]) -> None:
        env = self._env or {}
        run_rel = Path(os.path.relpath(str(self.run_dir), str(self.runs_root))).as_posix()
        gpu_names = env.get("torch_device_names") or env.get("gpu_names") or env.get("cuda_devices") or []
        if not gpu_names and isinstance(env.get("nvidia_smi"), list):
            gpu_names = [item.get("name") for item in env["nvidia_smi"] if isinstance(item, Mapping) and item.get("name")]
        if isinstance(gpu_names, str):
            gpu_names = [gpu_names]
        summary: Dict[str, Any] = {}
        if gates:
            summary = {
                "passed": bool(gates.get("passed", False)),
                "blocked": bool(gates.get("blocked", False)),
                "results": gates.get("results", []),
            }
        row = {
            "run_id": self.run_id,
            "paper": self.paper,
            "experiment": self.experiment,
            "seed": self.seed,
            "status": status,
            "run_dir": run_rel,
            "git_commit": env.get("git_commit"),
            "git_dirty": bool(env.get("git_dirty", False)),
            "host": env.get("hostname") or socket.gethostname(),
            "gpu_names": gpu_names,
            "started_at": _stamp(self._started_dt),
            "finished_at": _stamp(self._finished_dt),
            "tags": self.tags,
            "prereg_id": self.prereg.get("id") if self.prereg else None,
            "prereg_sha256": self.prereg_sha256,
            "gates": summary,
        }
        _append_jsonl(self.runs_root / "ledger.jsonl", row)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self._entered:
            return False
        self._finished_dt = _utc_now()
        started = self._started_dt or self._finished_dt
        wall_seconds = max(0.0, (self._finished_dt - started).total_seconds())
        # Always persist resource information and metrics, including failed runs.
        resources = self._resource_metrics(wall_seconds)
        self.set_metrics({"resources": resources})
        _json_dump(self.run_dir / "metrics.json", self._metrics)
        gates: Optional[Dict[str, Any]] = None
        if self.prereg is not None:
            try:
                from .prereg import evaluate_gates
                gates = evaluate_gates(self.prereg, self._metrics, experiment=self.experiment)
                _json_dump(self.run_dir / "gates.json", gates)
            except Exception as gate_exc:
                gates = {"passed": False, "blocked": False, "results": [{"error": str(gate_exc)}]}
                _json_dump(self.run_dir / "gates.json", gates)
        if exc_type is not None:
            status = "FAILED"
            self._status = status
            failure = {
                "status": status,
                "error": str(exc) if exc is not None else exc_type.__name__,
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
                "finished_at": _stamp(self._finished_dt),
            }
            _json_dump(self.run_dir / "FAILED.json", failure)
            (self.run_dir / "finished_at").write_text(_stamp(self._finished_dt) + "\n", encoding="utf-8")
            self._write_index(status, gates)
            return False
        blocked = bool(gates and gates.get("blocked"))
        if blocked:
            status = "BLOCKED"
            self._status = status
            _json_dump(self.run_dir / "BLOCKED.json", {"status": status, "gates": gates, "finished_at": _stamp(self._finished_dt)})
        else:
            status = "SUCCESS"
            self._status = status
            (self.run_dir / "_SUCCESS").write_text(_stamp(self._finished_dt) + "\n", encoding="utf-8")
        (self.run_dir / "finished_at").write_text(_stamp(self._finished_dt) + "\n", encoding="utf-8")
        self._write_index(status, gates)
        return False


__all__ = ["RunContext", "is_superseded"]



