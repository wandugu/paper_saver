"""Normalize original MNRE/Twitter2017 annotations without fabricated media."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from repro.p24.resources import sha256_file, write_json


def validate_image(path: str | Path) -> dict[str, Any]:
    from PIL import Image

    path = Path(path).expanduser().resolve()
    result: dict[str, Any] = {"path": str(path), "valid": False}
    if not path.is_file():
        return {**result, "reason": "missing"}
    result["bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    if not result["bytes"]:
        return {**result, "reason": "zero_byte"}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            result.update(width=width, height=height, source_mode=image.mode)
            if min(width, height) < 16:
                return {**result, "reason": "minimum_side_below_16"}
            rgb = image.convert("RGB")
            rgb.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return {**result, "reason": "decode_failure", "error": type(error).__name__}
    return {**result, "valid": True, "reason": None, "decoded_mode": "RGB"}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _span(value: Any, token_count: int, context: str) -> list[int]:
    if isinstance(value, dict):
        value = value.get("pos")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context}: expected half-open [start,end], got {value!r}")
    if any(type(index) is not int for index in value):
        raise ValueError(f"{context}: span indices must be integers")
    start, end = value
    if not 0 <= start < end <= token_count:
        raise ValueError(f"{context}: invalid half-open span {value!r} for {token_count} tokens")
    return [start, end]


def normalize_mnre_record(raw: dict[str, Any], *, split: str, index: int,
                          image_root: str | Path,
                          image_cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    tokens = raw.get("token", raw.get("tokens"))
    if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) for token in tokens):
        raise ValueError(f"{split}:{index}: tokens must be a nonempty string list")
    label = raw.get("relation", raw.get("label"))
    if not isinstance(label, str) or not label:
        raise ValueError(f"{split}:{index}: relation must be a nonempty string")
    head = _span(raw.get("h", raw.get("head")), len(tokens), f"{split}:{index}:head")
    tail = _span(raw.get("t", raw.get("tail")), len(tokens), f"{split}:{index}:tail")
    image_ids = raw.get("img_id", raw.get("images", []))
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    if not isinstance(image_ids, list) or any(not isinstance(image, str) for image in image_ids):
        raise ValueError(f"{split}:{index}: image IDs must be strings")
    image_root = Path(image_root).expanduser().resolve()
    cache = image_cache if image_cache is not None else {}
    images: list[str] = []
    original_indices: list[int] = []
    rejected: list[dict[str, Any]] = []
    for original_index, image_id in enumerate(image_ids):
        image = Path(image_id)
        if not image.is_absolute() and ".." in image.parts:
            raise ValueError(f"{split}:{index}: parent traversal in image ID")
        image = image.resolve() if image.is_absolute() else (image_root / image).resolve()
        key = str(image)
        if key not in cache:
            cache[key] = validate_image(image)
        if cache[key]["valid"]:
            images.append(key)
            original_indices.append(original_index)
        else:
            rejected.append({"original_index": original_index, "path": key,
                             "reason": cache[key]["reason"]})
    return {
        "id": f"mnre:{split}:{index:06d}", "tokens": tokens, "head": head, "tail": tail,
        "label": label, "images": images, "split": split,
        "source_image_ids": image_ids, "image_original_indices": original_indices,
        "invalid_images": rejected,
    }


def _read_mnre(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(line)
                except (SyntaxError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid annotation literal") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: annotation is not a mapping")
            yield line_number, raw


def _media_inventory(output: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entries = [cache[key] for key in sorted(cache)]
    _write_jsonl(output / "image_manifest.jsonl", entries)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    return {"manifest": str(output / "image_manifest.jsonl"),
            "manifest_sha256": sha256_file(output / "image_manifest.jsonl"),
            "content_tree_sha256": digest.hexdigest(), "unique_referenced_files": len(entries),
            "valid_files": sum(entry["valid"] for entry in entries),
            "valid_bytes": sum(entry.get("bytes", 0) for entry in entries if entry["valid"]),
            "invalid_reasons": dict(Counter(entry["reason"] for entry in entries if not entry["valid"]))}


def prepare_mnre(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Keep official ours_train/ours_val/ours_test membership and original images."""
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    split_sources = {"train": ("ours_train.txt", "train"),
                     "dev": ("ours_val.txt", "val"), "test": ("ours_test.txt", "test")}
    required = [root / "txt" / filename for filename, _ in split_sources.values()]
    required.append(root / "ours_rel2id.json")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    rel2id = json.loads(required[-1].read_text(encoding="utf-8"))
    if not isinstance(rel2id, dict) or not rel2id or any(
        not isinstance(label, str) or type(value) is not int for label, value in rel2id.items()
    ) or sorted(rel2id.values()) != list(range(len(rel2id))):
        raise ValueError("rel2id must map strings to unique contiguous integer IDs")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "rel2id.json", rel2id)
    image_cache: dict[str, dict[str, Any]] = {}
    splits: dict[str, Any] = {}
    content_keys: dict[str, set[str]] = {}
    combined_rows: list[dict[str, Any]] = []
    for split, (filename, image_directory) in split_sources.items():
        source = root / "txt" / filename
        rows = []
        for index, raw in _read_mnre(source):
            record = normalize_mnre_record(raw, split=split, index=index - 1,
                                          image_root=root / "img_org" / image_directory,
                                          image_cache=image_cache)
            if record["label"] not in rel2id:
                raise ValueError(f"{source}:{index}: unknown label {record['label']!r}")
            rows.append(record)
        _write_jsonl(output / f"{split}.jsonl", rows)
        combined_rows.extend(rows)
        content_keys[split] = {json.dumps([row["tokens"], row["head"], row["tail"]],
                                         ensure_ascii=False) for row in rows}
        splits[split] = {
            "records": len(rows), "source": str(source), "source_sha256": sha256_file(source),
            "normalized": str(output / f"{split}.jsonl"),
            "normalized_sha256": sha256_file(output / f"{split}.jsonl"),
            "valid_image_references": sum(len(row["images"]) for row in rows),
            "records_without_images": sum(not row["images"] for row in rows),
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        }
        print(f"MNRE {split}: {len(rows)} records; "
              f"{splits[split]['records_without_images']} without valid images", flush=True)
    combined = output.parent / "mnre.jsonl"
    _write_jsonl(combined, combined_rows)
    image_list = output / "image_files.txt"
    image_list.write_text("".join(
        str(Path(key).relative_to(root / "img_org")) + "\n"
        for key in sorted(image_cache) if image_cache[key]["valid"]
    ), encoding="utf-8")
    manifest = {
        "dataset": "MNRE", "task": "MRE", "schema_version": 1,
        "source_root": str(root), "source": "https://github.com/thecharm/MNRE",
        "version": "local official-named ours_* annotations; upstream commit unavailable",
        "official_splits_preserved": True, "span_convention": "zero_based_half_open",
        "source_rel2id": str(required[-1]), "source_rel2id_sha256": sha256_file(required[-1]),
        "rel2id": rel2id, "splits": splits, "media": _media_inventory(output, image_cache),
        "combined": str(combined), "combined_sha256": sha256_file(combined),
        "transfer": {"source_image_root": str(root / "img_org"),
                     "rsync_files_from": str(image_list),
                     "image_list_sha256": sha256_file(image_list),
                     "path_rewrite_required": "Rewrite images and invalid_images[].path after transfer; original manifest remains immutable"},
        "excluded_media": ["img_detect", "img_vg", "vg_data", "rel_1", "mre_*_dict.pth"],
        "media_policy": "Only img_org original attachments; no crop, placeholder or feature-file substitution",
        "cross_split_exact_pair_overlap": {
            f"{left}/{right}": len(content_keys[left] & content_keys[right])
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))},
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def bio_to_spans(tags: list[str]) -> list[list[Any]]:
    spans: list[list[Any]] = []
    start: int | None = None
    entity_type: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "O":
            prefix, current_type = "O", None
        elif "-" in tag and tag.split("-", 1)[0] in {"B", "I"}:
            prefix, current_type = tag.split("-", 1)
            if not current_type:
                raise ValueError(f"Empty entity type: {tag!r}")
        else:
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        continuation = prefix == "I" and start is not None and current_type == entity_type
        if start is not None and not continuation:
            spans.append([start, index, entity_type])
            start, entity_type = None, None
        if prefix != "O" and not continuation:
            start, entity_type = index, current_type
    return spans


def _read_twitter(path: Path) -> Iterable[dict[str, Any]]:
    image_id: str | None = None
    tokens: list[str] = []
    tags: list[str] = []
    for line_number, raw_line in enumerate([*path.read_text(encoding="utf-8").splitlines(), ""], start=1):
        line = raw_line.strip()
        if line.startswith("IMGID:"):
            if tokens:
                raise ValueError(f"{path}:{line_number}: missing post separator")
            image_id = line.removeprefix("IMGID:")
        elif not line:
            if tokens:
                if image_id is None:
                    raise ValueError(f"{path}:{line_number}: missing IMGID")
                yield {"image_id": image_id, "tokens": tokens, "tags": tags}
                image_id, tokens, tags = None, [], []
        else:
            fields = line.rsplit(None, 1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected token and BIO tag")
            tokens.append(fields[0])
            tags.append(fields[1])


def prepare_twitter2017(root: str | Path, output: str | Path, *,
                         image_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve() if image_root else root / "images"
    output.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    splits: dict[str, Any] = {}
    entity_types: set[str] = set()
    combined_rows: list[dict[str, Any]] = []
    for split, filename in (("train", "train.txt"), ("dev", "valid.txt"), ("test", "test.txt")):
        source = root / filename
        rows = []
        for index, post in enumerate(_read_twitter(source)):
            spans = bio_to_spans(post["tags"])
            entity_types.update(span[2] for span in spans)
            image_id = post["image_id"]
            if Path(image_id).is_absolute() or ".." in Path(image_id).parts:
                raise ValueError(f"Unsafe Twitter2017 image ID: {image_id!r}")
            image = image_root / (image_id if Path(image_id).suffix else image_id + ".jpg")
            key = str(image.resolve())
            if key not in cache:
                cache[key] = validate_image(image)
            valid = cache[key]["valid"]
            rows.append({"id": f"twitter2017:{split}:{index:06d}", "tokens": post["tokens"],
                         "spans": spans, "images": [key] if valid else [], "split": split,
                         "source_image_ids": [image_id], "image_original_indices": [0] if valid else [],
                         "invalid_images": [] if valid else [{"original_index": 0, "path": key,
                                                               "reason": cache[key]["reason"]}]})
        _write_jsonl(output / f"{split}.jsonl", rows)
        combined_rows.extend(rows)
        splits[split] = {"records": len(rows), "source": str(source),
                         "source_sha256": sha256_file(source),
                         "normalized": str(output / f"{split}.jsonl"),
                         "normalized_sha256": sha256_file(output / f"{split}.jsonl"),
                         "spans": sum(len(row["spans"]) for row in rows),
                         "records_without_images": sum(not row["images"] for row in rows)}
    combined = output.parent / "twitter2017.jsonl"
    _write_jsonl(combined, combined_rows)
    manifest = {"dataset": "Twitter2017", "task": "MNER", "schema_version": 1,
                "source": "https://github.com/jefferyYu/UMT", "source_root": str(root),
                "version": "local UMT annotation release; upstream commit unavailable",
                "official_splits_preserved": True, "span_convention": "zero_based_half_open",
                "bio_policy": "IOB2; orphan or type-changing I starts a span; raw annotations unchanged",
                "splits": splits, "entity_types": sorted(entity_types),
                "combined": str(combined), "combined_sha256": sha256_file(combined),
                "media": _media_inventory(output, cache),
                "multimodal_usable": any(entry["valid"] for entry in cache.values()),
                "prepared_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("mnre", "twitter2017"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    args = parser.parse_args()
    if args.task == "mnre":
        prepare_mnre(args.root, args.output)
    else:
        prepare_twitter2017(args.root, args.output, image_root=args.image_root)


if __name__ == "__main__":
    main()
