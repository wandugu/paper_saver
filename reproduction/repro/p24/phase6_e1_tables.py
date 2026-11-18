"""Post hoc E1 table specs; invoke through python -m repro.make_tables."""

import csv
import json
import os
from pathlib import Path

from repro.common.tables import ColumnSpec, RowSpec, TableSpec, build_table


ROOT = Path(os.environ.get("P24_PHASE6_RUNS", Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase6"))
LEDGER = [json.loads(line) for line in (ROOT / "ledger.jsonl").read_text().splitlines() if line.strip()]
RUNS = [row for row in LEDGER if row["experiment"] == "phase6_e1" and row["status"] == "SUCCESS"]
if len(RUNS) != 1:
    raise ValueError("Expected exactly one completed E1 run")
SELECT = {"run_id": RUNS[0]["run_id"]}
TABLES = []


def one_table(name, caption, rows, columns):
    name = f"p24_phase6_E1{name}"
    generated = []
    for label, prefix in rows:
        spec = TableSpec(name, caption, [RowSpec(label, SELECT)],
                         [ColumnSpec(header, f"{prefix}.{key}", fmt, "single") for header, key, fmt in columns])
        generated.append(build_table(spec, ROOT))
    out = Path(os.environ["P24_PHASE6_TABLES"])
    out.mkdir(parents=True, exist_ok=True)
    first = generated[0]
    md_header = first["markdown"].splitlines()[:-1]
    tex_header = first["latex"].splitlines()[:-3]
    tex_end = first["latex"].splitlines()[-2:]
    markdown = [*md_header, *(item["markdown"].splitlines()[-1] for item in generated)]
    latex = [f"% {caption}", *tex_header, *(item["latex"].splitlines()[-3] for item in generated), *tex_end]
    table = {"id": name, "caption": caption, "headers": first["headers"], "rows": [item["rows"][0] for item in generated]}
    for extension, content in (("md", "\n".join(markdown)), ("tex", "\n".join(latex)),
                               ("json", json.dumps(table, indent=2, ensure_ascii=False))):
        with (out / f"{name}.{extension}").open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    with (out / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row", *table["headers"], "run_id"])
        for row in table["rows"]:
            writer.writerow([row["label"], *(cell["value"] for cell in row["cells"]), *row["run_ids"]])


FIVE = "{:.5f}"
INTEGER = "{:.0f}"
BOOT = [("F1 delta", "point", FIVE), ("95% low", "low", FIVE), ("95% high", "high", FIVE),
        ("One-sided 95% upper", "upper", FIVE), ("Bootstrap SE", "se", FIVE), ("MDE80", "mde80", FIVE),
        ("Posts", "posts", INTEGER)]
for split in ("dev", "test"):
    phase = "development (also used for checkpoint selection)" if split == "dev" else "previously viewed test, descriptive only"
    root = f"E1a.{split}"
    one_table(f"a_{split}", f"Post hoc A6 always-vision minus text-only on {phase}; post bootstrap conditional on fitted models.",
              [(f"seed{seed}", f"{root}.seeds.{seed}") for seed in (42, 43, 44)] + [("shared post draws", f"{root}.pooled")],
              [("Text F1", "text_f1", FIVE), ("Vision F1", "vision_f1", FIVE)] + BOOT)
    one_table(f"a_{split}_seed_t", f"Post hoc A6 on {phase}; training-seed t interval, not the post bootstrap.",
              [("training seeds", f"{root}.seed_t")],
              [("Mean F1 delta", "point", FIVE), ("Seed SD", "sd", FIVE), ("95% t low", "low", FIVE),
               ("95% t high", "high", FIVE), ("One-sided 95% t upper", "upper", FIVE), ("Seeds", "n", INTEGER)])

one_table("b", "Post hoc calibration: paired post AUROC difference, text uncertainty minus learned gate. Seed means are descriptive.",
          [(f"{study} {seed}", f"E1b.{study}.{seed}") for study in ("A1", "A2") for seed in (42, 43, 44, "mean")],
          [("Gate AUROC", "saver", FIVE), ("Text AUROC", "uncertainty", FIVE), ("Delta AUROC", "point", FIVE),
           ("95% low", "low", FIVE), ("95% high", "high", FIVE), ("Posts", "posts", INTEGER),
           ("Invalid draws", "invalid_draws", INTEGER)])
one_table("c", "Post hoc post-level signs (two-sided exact binomial). A6 test and all pooled-count p values are descriptive; seeds are not independent.",
          [(f"{study} {seed}", f"E1c.{study}.{seed}") for study in ("A6_dev", "A6_test", "A1", "A2")
           for seed in (42, 43, 44, "pooled")],
          [("Fix", "fix", INTEGER), ("Break", "break", INTEGER), ("Zero", "zero", INTEGER), ("p", "p", "{:.7g}")])
