"""Build deterministic tables from completed reproducibility runs.

The table layer deliberately consumes only ``ledger.jsonl`` and each run's
``metrics.json``.  It does not contain paper constants or fallback values.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ledger import is_superseded


@dataclass(frozen=True)
class ColumnSpec:
    header: str
    metric: str
    fmt: str = "{:.4f}"
    agg: str = "mean_sd"

    def __post_init__(self) -> None:
        if self.agg not in {"mean_sd", "mean", "single", "count"}:
            raise ValueError(f"unsupported aggregation: {self.agg}")


@dataclass(frozen=True)
class RowSpec:
    label: str
    select: dict[str, Any]


@dataclass(frozen=True)
class TableSpec:
    id: str
    caption: str
    rows: list[RowSpec]
    columns: list[ColumnSpec]


def _read_ledger(runs_root: Path) -> list[dict[str, Any]]:
    path = runs_root / "ledger.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except OSError:
        return []
    return records


def _dot_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split(".") if path else []:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _matches(record: dict[str, Any], select: dict[str, Any]) -> bool:
    """Match a ledger record against row selectors.

    Selectors may address top-level fields (``paper``, ``experiment``,
    ``status``), ``tags`` members through either ``tags.foo`` or ``foo``, and
    may use a sequence to express membership.  ``allow_blocked`` is a control
    key and is not itself matched.
    """

    for key, wanted in (select or {}).items():
        if key in {"allow_blocked", "include_blocked"}:
            continue
        actual = _dot_get(record, key)
        if actual is None and not key.startswith("tags."):
            actual = _dot_get(record.get("tags", {}), key)
        if isinstance(wanted, (list, tuple, set)):
            if actual not in wanted:
                return False
        elif isinstance(wanted, dict):
            # Tag selectors are naturally partial (``{"tags": {"split":
            # "test"}}``); other dictionary-valued fields use exact matching.
            if key == "tags" and isinstance(actual, dict):
                if any(actual.get(k) != v for k, v in wanted.items()):
                    return False
            elif actual != wanted:
                return False
        elif actual != wanted:
            return False
    return True


def _format_value(value: float, fmt: str) -> str:
    try:
        return fmt.format(value)
    except Exception:
        try:
            return format(value, fmt)
        except Exception:
            return str(value)


def _aggregate(column: ColumnSpec, values: list[tuple[Any, str]]) -> dict[str, Any]:
    run_ids = sorted(run_id for _, run_id in values)
    usable = [v for v, _ in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not usable:
        reason = "no run" if not values else "missing"
        return {"value": f"n/a ({reason})", "run_ids": run_ids, "source_run_ids": run_ids, "n": 0}

    n = len(usable)
    if column.agg == "count":
        rendered = str(n)
        data: dict[str, Any] = {"value": rendered, "run_ids": run_ids, "source_run_ids": run_ids, "n": n, "mean": n}
    elif column.agg == "single":
        rendered = _format_value(float(usable[0]), column.fmt)
        data = {"value": rendered, "run_ids": run_ids, "source_run_ids": run_ids, "n": n, "mean": usable[0]}
    elif column.agg == "mean":
        mean = statistics.fmean(usable)
        rendered = _format_value(mean, column.fmt)
        data = {"value": rendered, "run_ids": run_ids, "source_run_ids": run_ids, "n": n, "mean": mean}
    else:  # mean_sd
        mean = statistics.fmean(usable)
        # A zero SD for a singleton makes the uncertainty explicit and keeps
        # rendering deterministic across one and many seeds.
        sd = statistics.stdev(usable) if n > 1 else 0.0
        rendered = f"{_format_value(mean, column.fmt)} ± {_format_value(sd, column.fmt)} (n={n})"
        data = {"value": rendered, "run_ids": run_ids, "source_run_ids": run_ids, "n": n, "mean": mean, "sd": sd}
    return data


def build_table(spec: TableSpec, runs_root: str | Path) -> dict[str, Any]:
    """Build one table and return structured cells plus Markdown/LaTeX text."""

    root = Path(runs_root)
    ledger = _read_ledger(root)
    table_rows: list[dict[str, Any]] = []

    for row_spec in spec.rows:
        select = dict(row_spec.select or {})
        allow_blocked = bool(select.get("allow_blocked") or select.get("include_blocked"))
        explicit_status = select.get("status")
        selected: list[dict[str, Any]] = []
        for rec in ledger:
            status = rec.get("status")
            # Successful runs are the default population.  A selector may
            # explicitly opt in to BLOCKED runs, but failed runs never enter
            # a reported table.
            blocked_allowed = allow_blocked or explicit_status == "BLOCKED" or (
                isinstance(explicit_status, (list, tuple, set))
                and "BLOCKED" in explicit_status
            )
            if status != "SUCCESS" and not (status == "BLOCKED" and blocked_allowed):
                continue
            rel = rec.get("run_dir")
            if is_superseded(root / str(rel) if rel else root / str(rec.get("paper", "")) / str(rec.get("run_id", ""))):
                continue
            if _matches(rec, select):
                selected.append(rec)
        selected.sort(key=lambda r: str(r.get("run_id", "")))

        cells: list[dict[str, Any]] = []
        for col in spec.columns:
            vals: list[tuple[Any, str]] = []
            for rec in selected:
                run_id = str(rec.get("run_id", ""))
                rel = rec.get("run_dir")
                run_dir = root / str(rel) if rel else root / str(rec.get("paper", "")) / run_id
                metrics_path = run_dir / "metrics.json"
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}
                vals.append((_dot_get(metrics, col.metric), run_id))
            cells.append(_aggregate(col, vals))
        table_rows.append({"label": row_spec.label, "cells": cells, "run_ids": sorted({rid for c in cells for rid in c["run_ids"]})})

    headers = [c.header for c in spec.columns]
    md_lines = [f"**{spec.caption}**", "", "| " + " | ".join([""] + headers) + " |", "| " + " | ".join(["---"] + ["---"] * len(headers)) + " |"]
    for row in table_rows:
        md_lines.append("| " + " | ".join([row["label"]] + [c["value"] for c in row["cells"]]) + " |")
    markdown = "\n".join(md_lines) + "\n"

    def _latex_escape(text: str) -> str:
        return str(text).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")

    spec_cols = "l" + "".join("c" for _ in spec.columns)
    latex_lines = [
        f"\\begin{{tabular}}{{{spec_cols}}}",
        "\\toprule",
        " & ".join([_latex_escape("")] + [_latex_escape(h) for h in headers]) + " \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        body = " & ".join([_latex_escape(row["label"])] + [_latex_escape(c["value"]) for c in row["cells"]])
        ids = sorted(row.get("run_ids", []))
        comment = f" % runs: {','.join(ids)}" if ids else " % runs:"
        latex_lines.append(body + " \\\\" + comment)
    latex_lines += ["\\bottomrule", "\\end{tabular}"]
    latex = "\n".join(latex_lines) + "\n"

    return {
        "id": spec.id,
        "caption": spec.caption,
        "headers": headers,
        "rows": table_rows,
        "markdown": markdown,
        "latex": latex,
    }


__all__ = ["ColumnSpec", "RowSpec", "TableSpec", "build_table"]
