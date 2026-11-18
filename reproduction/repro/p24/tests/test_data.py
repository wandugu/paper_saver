import json
from pathlib import Path

import pytest
from PIL import Image

from repro.p24.data import (bio_to_spans, load_records, normalize_mnre_record,
                           prepare_mnre, prepare_twitter2017, validate_image)
from repro.p24.resources import file_inventory, sha256_file


def make_image(path: Path, size: tuple[int, int] = (20, 20), mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


def raw_record(image: str = "real.png") -> dict:
    return {"token": ["Ann", "in", "Paris"], "h": {"pos": [0, 1]},
            "t": {"pos": [2, 3]}, "relation": "None", "img_id": image}


def test_validation_checks_verify_rgb_minimum_and_missing(tmp_path):
    make_image(tmp_path / "gray.png", mode="L")
    make_image(tmp_path / "tiny.png", size=(15, 20))
    (tmp_path / "empty.jpg").touch()
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    assert validate_image(tmp_path / "gray.png")["decoded_mode"] == "RGB"
    assert validate_image(tmp_path / "tiny.png")["reason"] == "minimum_side_below_16"
    assert validate_image(tmp_path / "empty.jpg")["reason"] == "zero_byte"
    assert validate_image(tmp_path / "bad.jpg")["reason"] == "decode_failure"
    assert validate_image(tmp_path / "absent.jpg")["reason"] == "missing"


def test_normalization_preserves_half_open_and_attachment_order(tmp_path):
    make_image(tmp_path / "first.png")
    make_image(tmp_path / "third.png")
    record = raw_record()
    record["img_id"] = ["first.png", "missing.png", "third.png"]
    normalized = normalize_mnre_record(record, split="train", index=5, image_root=tmp_path)
    assert normalized["head"] == [0, 1]
    assert normalized["tail"] == [2, 3]
    assert normalized["image_original_indices"] == [0, 2]
    assert normalized["images"] == [str(tmp_path / "first.png"), str(tmp_path / "third.png")]
    assert normalized["invalid_images"][0]["original_index"] == 1


@pytest.mark.parametrize("span", [[-1, 1], [1, 1], [0, 4], [False, 1], [0.0, 1]])
def test_bad_spans_rejected(tmp_path, span):
    record = raw_record()
    record["h"]["pos"] = span
    with pytest.raises(ValueError, match="span"):
        normalize_mnre_record(record, split="train", index=0, image_root=tmp_path)


def test_no_image_never_substitutes_inf_placeholder(tmp_path):
    make_image(tmp_path / "inf.png")
    normalized = normalize_mnre_record(raw_record("missing.jpg"), split="train", index=0,
                                        image_root=tmp_path)
    assert normalized["images"] == []
    assert normalized["invalid_images"][0]["reason"] == "missing"


def test_parent_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="traversal"):
        normalize_mnre_record(raw_record("../outside.jpg"), split="train", index=0,
                              image_root=tmp_path)


def test_official_split_and_rel2id_preserved(tmp_path):
    source, output = tmp_path / "source", tmp_path / "normalized"
    (source / "txt").mkdir(parents=True)
    (source / "ours_rel2id.json").write_text(json.dumps({"None": 0}))
    for split in ("train", "val", "test"):
        make_image(source / "img_org" / split / "real.png")
        (source / "txt" / f"ours_{split}.txt").write_text(repr(raw_record()) + "\n")
    manifest = prepare_mnre(source, output)
    assert manifest["official_splits_preserved"]
    assert manifest["splits"]["dev"]["source"].endswith("ours_val.txt")
    assert manifest["media"]["valid_files"] == 3
    assert load_records(output / "test.jsonl")[0]["split"] == "test"
    assert len(load_records(output.parent / "mnre.jsonl")) == 3
    assert (output / "image_files.txt").read_text().splitlines() == [
        "test/real.png", "train/real.png", "val/real.png"]
    assert json.loads((output / "rel2id.json").read_text()) == {"None": 0}
    assert sha256_file(source / "txt" / "ours_train.txt") == manifest["splits"]["train"]["source_sha256"]


def test_mnre_literal_is_not_executed(tmp_path):
    source = tmp_path / "source"
    (source / "txt").mkdir(parents=True)
    (source / "ours_rel2id.json").write_text('{"None": 0}')
    for split in ("train", "val", "test"):
        (source / "txt" / f"ours_{split}.txt").write_text("__import__('os').system('false')\n")
    with pytest.raises(ValueError, match="invalid annotation"):
        prepare_mnre(source, tmp_path / "out")


def test_bio_half_open_and_orphan_handling():
    assert bio_to_spans(["B-PER", "I-PER", "O", "I-LOC", "I-ORG"]) == [
        [0, 2, "PER"], [3, 4, "LOC"], [4, 5, "ORG"]]
    with pytest.raises(ValueError, match="BIO"):
        bio_to_spans(["INVALID"])


def test_twitter2017_annotation_only_is_not_multimodal(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "valid", "test"):
        (source / f"{split}.txt").write_text("IMGID:tweet1\nAnn\tB-PER\nis\tO\nhere\tO\n\n")
    output = tmp_path / "normalized"
    manifest = prepare_twitter2017(source, output)
    assert not manifest["multimodal_usable"]
    assert len(load_records(output.parent / "twitter2017.jsonl")) == 3
    assert load_records(output / "dev.jsonl")[0]["spans"] == [[0, 1, "PER"]]
    assert manifest["splits"]["train"]["records_without_images"] == 1


def test_file_inventory_is_stable_and_ignores_partial(tmp_path):
    (tmp_path / "weights").write_bytes(b"verified")
    (tmp_path / "weights.partial").write_bytes(b"incomplete")
    first = file_inventory(tmp_path)
    assert first == file_inventory(tmp_path)
    assert [entry["path"] for entry in first["files"]] == ["weights"]
