from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from processor.saver_dataset import SaverCollator, SaverJsonlDataset


class FakeBatchEncoding(dict):
    def __init__(self, values, word_ids):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self, batch_index):
        return self._word_ids[batch_index]


class FakeTokenizer:
    def __call__(
        self,
        texts,
        is_split_into_words,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        rows = []
        masks = []
        all_word_ids = []
        longest = min(max(len(tokens) + 2 for tokens in texts), max_length)
        for tokens in texts:
            visible = tokens[: max(0, max_length - 2)]
            ids = [1] + list(range(3, 3 + len(visible))) + [2]
            word_ids = [None] + list(range(len(visible))) + [None]
            mask = [1] * len(ids)
            while len(ids) < longest:
                ids.append(0)
                mask.append(0)
                word_ids.append(None)
            rows.append(ids)
            masks.append(mask)
            all_word_ids.append(word_ids)
        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(rows),
                "attention_mask": torch.tensor(masks),
            },
            all_word_ids,
        )


class FakeImageProcessor:
    def __call__(self, images, max_num_patches, return_tensors):
        image_count = len(images)
        return {
            "pixel_values": torch.ones(
                image_count,
                max_num_patches,
                6,
            ),
            "pixel_attention_mask": torch.ones(
                image_count,
                max_num_patches,
                dtype=torch.bool,
            ),
            "spatial_shapes": torch.tensor(
                [[1, max_num_patches]] * image_count,
                dtype=torch.long,
            ),
        }


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ["a.png", "b.png", "c.png"]:
            Image.new("RGB", (4, 4), color=(20, 30, 40)).save(self.root / name)

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_mre_multi_image_batch(self):
        path = self._write(
            "mre.jsonl",
            [
                {
                    "id": "r1",
                    "tokens": ["Ada", "works", "at", "OpenAI"],
                    "images": ["a.png", "b.png"],
                    "head": [0, 1],
                    "tail": [3, 4],
                    "relation": "works_for",
                },
                {
                    "id": "r2",
                    "tokens": ["Bob", "visits", "Paris"],
                    "images": ["c.png"],
                    "head": [0, 1],
                    "tail": [2, 3],
                    "relation": "visits",
                },
            ],
        )
        dataset = SaverJsonlDataset(
            str(path),
            "mre",
            {"works_for": 0, "visits": 1},
        )
        collator = SaverCollator(
            FakeTokenizer(),
            FakeImageProcessor(),
            "mre",
            screen_max_num_patches=2,
            detail_max_num_patches=5,
        )
        batch = collator([dataset[0], dataset[1]])
        self.assertEqual(batch["image_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["global_inputs"]["pixel_values"].shape, (2, 2, 2, 6))
        self.assertEqual(batch["detail_inputs"]["pixel_values"].shape, (2, 2, 5, 6))
        self.assertEqual(batch["head_spans"].tolist(), [[1, 2], [1, 2]])
        self.assertEqual(batch["tail_spans"].tolist(), [[4, 5], [3, 4]])

    def test_mner_enumerates_spans_and_exact_labels(self):
        path = self._write(
            "mner.jsonl",
            [
                {
                    "id": "n1",
                    "tokens": ["New", "York", "welcomes", "Ada"],
                    "images": ["a.png", "b.png", "c.png"],
                    "entities": [
                        {"start": 0, "end": 2, "type": "LOC"},
                        {"start": 3, "end": 4, "type": "PER"},
                    ],
                }
            ],
        )
        dataset = SaverJsonlDataset(
            str(path),
            "mner",
            {"NONE": 0, "PER": 1, "LOC": 2},
        )
        collator = SaverCollator(
            FakeTokenizer(),
            FakeImageProcessor(),
            "mner",
            max_span_width=2,
            screen_max_num_patches=2,
            detail_max_num_patches=4,
        )
        batch = collator([dataset[0]])
        valid_word_spans = batch["word_spans"][0][batch["span_mask"][0]]
        valid_labels = batch["labels"][0][batch["span_mask"][0]]
        label_by_span = {
            tuple(span.tolist()): int(label)
            for span, label in zip(valid_word_spans, valid_labels)
        }
        self.assertEqual(label_by_span[(0, 2)], 2)
        self.assertEqual(label_by_span[(3, 4)], 1)
        self.assertEqual(label_by_span[(1, 2)], 0)

    def test_mner_rejects_unrepresentable_or_overlapping_gold(self):
        overlapping_path = self._write(
            "overlap.jsonl",
            [
                {
                    "id": "overlap",
                    "tokens": ["New", "York", "City"],
                    "images": ["a.png"],
                    "entities": [
                        {"start": 0, "end": 2, "type": "LOC"},
                        {"start": 1, "end": 3, "type": "LOC"},
                    ],
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "overlapping entities"):
            SaverJsonlDataset(
                str(overlapping_path),
                "mner",
                {"NONE": 0, "LOC": 1},
            )

        wide_path = self._write(
            "wide.jsonl",
            [
                {
                    "id": "wide",
                    "tokens": ["New", "York", "City"],
                    "images": ["a.png"],
                    "entities": [
                        {"start": 0, "end": 3, "type": "LOC"},
                    ],
                }
            ],
        )
        dataset = SaverJsonlDataset(
            str(wide_path),
            "mner",
            {"NONE": 0, "LOC": 1},
        )
        collator = SaverCollator(
            FakeTokenizer(),
            FakeImageProcessor(),
            "mner",
            max_span_width=2,
            screen_max_num_patches=2,
            detail_max_num_patches=4,
        )
        with self.assertRaisesRegex(ValueError, "wider than max_span_width"):
            collator([dataset[0]])


if __name__ == "__main__":
    unittest.main()
