"""Pre-registration loading and gate evaluation."""

from __future__ import annotations

from hashlib import sha256
import operator
from pathlib import Path

from .paths import expand_config_paths

try:
    import yaml
except Exception:  # pragma: no cover - dependency supplied by project
    yaml = None


_OPS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def load_prereg_with_sha256(path) -> tuple[dict, str]:
    """Parse and hash the same snapshot of a registration file."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load pre-registration YAML")
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pre-registration document must be a mapping")
    for field in ("id", "paper", "gates"):
        if field not in data:
            raise ValueError(f"pre-registration missing required field: {field}")
    if not isinstance(data["gates"], list):
        raise ValueError("pre-registration gates must be a list")
    for index, gate in enumerate(data["gates"]):
        if not isinstance(gate, dict):
            raise ValueError(f"gate {index} must be a mapping")
        for field in ("metric", "op", "value"):
            if field not in gate:
                raise ValueError(f"gate {index} missing required field: {field}")
        if gate["op"] not in _OPS:
            raise ValueError(f"gate {index} has unsupported op: {gate['op']!r}")
    return expand_config_paths(data), sha256(raw).hexdigest()


def load_prereg(path) -> dict:
    return load_prereg_with_sha256(path)[0]


def require_fixed_prereg(ctx, path) -> dict:
    """Verify the run used the exact required template, including its resolved fields."""
    expected, expected_sha256 = load_prereg_with_sha256(path)
    if ctx.prereg_sha256 != expected_sha256 or ctx.prereg != expected:
        raise ValueError(f"Run requires the exact preregistration {Path(path).name} (SHA-256 {expected_sha256})")
    return ctx.prereg


def _lookup(metrics, path):
    current = metrics
    for part in str(path).split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None, False
    return current, True


def evaluate_gates(prereg: dict, metrics: dict, experiment: str | None = None) -> dict:
    """Evaluate gates; a gate with an ``experiment`` field only applies to runs of that experiment."""
    results = []
    blocked = False
    for index, gate in enumerate(prereg.get("gates", [])):
        scope = gate.get("experiment")
        if scope is not None and experiment is not None and scope != experiment:
            continue
        name = gate.get("name", f"gate_{index}")
        metric_path = gate.get("metric")
        actual, present = _lookup(metrics, metric_path)
        if not present:
            passed = False
            result = {
                "name": name,
                "metric": metric_path,
                "op": gate.get("op"),
                "expected": gate.get("value"),
                "passed": False,
                "missing": True,
            }
        else:
            try:
                passed = bool(_OPS[gate["op"]](actual, gate["value"]))
            except Exception as exc:
                passed = False
                result = {
                    "name": name,
                    "metric": metric_path,
                    "op": gate.get("op"),
                    "expected": gate.get("value"),
                    "actual": actual,
                    "passed": False,
                    "error": str(exc),
                }
            else:
                result = {
                    "name": name,
                    "metric": metric_path,
                    "op": gate.get("op"),
                    "expected": gate.get("value"),
                    "actual": actual,
                    "passed": passed,
                }
        if not passed and str(gate.get("on_fail", "WARN")).upper() == "BLOCK":
            blocked = True
        results.append(result)
    return {"passed": all(item["passed"] for item in results), "blocked": blocked, "results": results}


__all__ = ["load_prereg", "load_prereg_with_sha256", "require_fixed_prereg", "evaluate_gates"]

