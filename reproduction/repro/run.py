"""Run a registered reproduction experiment under :class:`RunContext`."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml

from .common.ledger import RunContext
from .common.paths import expand_config_paths
from .common.prereg import load_prereg_with_sha256


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        obj = yaml.safe_load(fh)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return expand_config_paths(obj)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a reproducibility experiment")
    parser.add_argument("paper")
    parser.add_argument("experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--prereg", type=Path, default=None)
    parser.add_argument("--expect-prereg-sha256", help="Require this SHA-256 for the actual registration bytes")
    args = parser.parse_args(argv)

    try:
        config = _read_config(args.config)
    except Exception as exc:
        parser.error(f"cannot read config: {exc}")
        return 2

    module_name = f"repro.{args.paper.lower()}.{args.experiment}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only turn a missing runner itself into the friendly diagnostic; an
        # import error inside an existing runner should retain its traceback.
        if exc.name == module_name or (exc.name and exc.name.startswith(f"repro.{args.paper.lower()}")):
            parser.error(f"experiment module not found: {module_name}")
            return 2
        raise
    except ImportError as exc:
        parser.error(f"could not import experiment module {module_name}: {exc}")
        return 2

    runner = getattr(module, "main", None)
    if not callable(runner):
        parser.error(f"experiment module {module_name} does not define callable main(ctx, config)")
        return 2

    prereg: Any = None
    prereg_sha256 = None
    if args.prereg is not None:
        try:
            prereg, prereg_sha256 = load_prereg_with_sha256(args.prereg)
        except Exception as exc:
            parser.error(f"cannot load preregistration: {exc}")
    if args.expect_prereg_sha256 is not None and args.expect_prereg_sha256 != prereg_sha256:
        parser.error(f"preregistration SHA-256 mismatch (read {prereg_sha256 or 'no registration'})")

    kwargs = {"paper": args.paper.upper(), "experiment": args.experiment, "config": config, "seed": args.seed}
    if args.runs_root is not None:
        kwargs["runs_root"] = args.runs_root
    if prereg is not None:
        kwargs["prereg"] = prereg
        kwargs["prereg_sha256"] = prereg_sha256
    try:
        with RunContext(**kwargs) as ctx:
            result = runner(ctx, config)
            # Metrics are intentionally written only through RunContext.  A
            # runner's return value is ignored by design.
        return 0
    except Exception:
        # RunContext records FAILED.json and re-raises.  Keep the CLI's normal
        # traceback so batch jobs retain the actual failure cause.
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

