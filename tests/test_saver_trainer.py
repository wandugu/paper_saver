from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from models.saver import SaverForMNER, SaverOutput
from modules.saver_trainer import SaverTrainer, TrainingConfig


class FakeConfig:
    gate_threshold = 0.5
    gate_warmup_steps = 0

    @staticmethod
    def to_dict():
        return {
            "gate_threshold": 0.5,
            "gate_warmup_steps": 0,
        }


class FakeMnerModel(nn.Module):
    decode_non_overlapping = staticmethod(SaverForMNER.decode_non_overlapping)

    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.head = nn.Parameter(torch.tensor(0.25))

    def forward(self, labels, span_mask, threshold=0.5, **kwargs):
        safe_labels = labels.clamp_min(0)
        logits = F.one_hot(safe_labels, num_classes=3).float() * 5.0
        logits = logits + self.head * 0.0
        gate_probability = torch.full(
            labels.shape,
            0.10,
            device=labels.device,
        )
        gate_probability[:, 0] = 0.90
        gate_active = (gate_probability >= float(threshold)) & span_mask.bool()
        batch_size, span_count = labels.shape
        return SaverOutput(
            loss=logits.sum() * 0.0,
            logits=logits,
            text_logits=logits,
            visual_logits=logits,
            gate_probability=gate_probability,
            gate_active=gate_active,
            selected_indices=torch.zeros(
                batch_size,
                span_count,
                1,
                dtype=torch.long,
                device=labels.device,
            ),
            attachment_relevance=torch.zeros(
                batch_size,
                span_count,
                1,
                device=labels.device,
            ),
            selected_regions=torch.full(
                (batch_size, span_count, 2, 2),
                -1,
                dtype=torch.long,
                device=labels.device,
            ),
        )


class TrainerTest(unittest.TestCase):
    def test_mner_metrics_mask_padding_and_write_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SaverTrainer(
                FakeMnerModel(),
                "mner",
                TrainingConfig(
                    epochs=1,
                    bf16=False,
                    output_dir=temporary,
                ),
                device=torch.device("cpu"),
            )
            batch = {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[True, True]]),
                "global_inputs": {},
                "detail_inputs": {},
                "image_mask": torch.tensor([[True]]),
                "labels": torch.tensor([[1, -100]]),
                "spans": torch.tensor([[[0, 1], [1, 2]]]),
                "word_spans": torch.tensor([[[0, 1], [1, 2]]]),
                "span_widths": torch.tensor([[1, 1]]),
                "span_mask": torch.tensor([[True, False]]),
                "sample_ids": ["sample-1"],
                "image_paths": [["image.png"]],
            }
            metrics = trainer.evaluate(
                [batch],
                artifact_name="test",
            )

            self.assertEqual(metrics["valid_unit_count"], 1.0)
            self.assertEqual(metrics["activation_coverage"], 1.0)
            self.assertAlmostEqual(metrics["mean_gate_score"], 0.9, places=6)
            self.assertEqual(metrics["all_on_rate"], 1.0)
            self.assertEqual(metrics["strict_entity_f1"], 1.0)

            prediction_path = Path(temporary) / "test_predictions.jsonl"
            record = json.loads(prediction_path.read_text(encoding="utf-8"))
            self.assertEqual(record["sample_id"], "sample-1")
            self.assertEqual(len(record["spans"]), 1)

            checkpoint = trainer.save_checkpoint("probe")
            original = trainer.model.head.detach().clone()
            trainer.model.head.data.fill_(9.0)
            trainer.restore_checkpoint(checkpoint)
            self.assertTrue(torch.equal(trainer.model.head, original))


if __name__ == "__main__":
    unittest.main()
