"""Command line interface for rendering tables from the run ledger."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from .common.tables import TableSpec, build_table


def _load_spec(path: str | Path) -> Any:
    spec_path = Path(path).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"table spec not found: {spec_path}")
    module_name = f"repro_table_spec_{abs(hash(str(spec_path)))}"
    module_spec = importlib.util.spec_from_file_location(module_name, spec_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load table spec: {spec_path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    if not hasattr(module, "TABLES"):
        raise AttributeError(f"table spec {spec_path} must define TABLES")
    return module.TABLES


def _coerce_table(obj: Any) -> TableSpec:
    if isinstance(obj, TableSpec):
        return obj
    if isinstance(obj, dict):
        from .common.tables import ColumnSpec, RowSpec

        rows = [r if isinstance(r, RowSpec) else RowSpec(**r) for r in obj.get("rows", [])]
        columns = [c if isinstance(c, ColumnSpec) else ColumnSpec(**c) for c in obj.get("columns", [])]
        return TableSpec(id=obj["id"], caption=obj.get("caption", obj["id"]), rows=rows, columns=columns)
    raise TypeError(f"TABLES entries must be TableSpec or dict, got {type(obj)!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Markdown and LaTeX tables from repro runs")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        tables = [_coerce_table(item) for item in _load_spec(args.spec)]
    except Exception as exc:
        parser.error(str(exc))
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for table in sorted(tables, key=lambda t: t.id):
        result = build_table(table, args.runs_root)
        (args.out_dir / f"{table.id}.md").write_text(result["markdown"], encoding="utf-8")
        (args.out_dir / f"{table.id}.tex").write_text(result["latex"], encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

