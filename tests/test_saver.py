from __future__ import annotations

import unittest

import torch
from torch import nn

from models.saver import (
    RiskControlledCalibrator,
    SaverConfig,
    SaverForMNER,
    SaverForMRE,
    SaverRouter,
)


class DummyTextEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.embedding(input_ids) * attention_mask.unsqueeze(-1)


class DummyVisionEncoder(nn.Module):
    def __init__(self, patch_size: int = 6, hidden_size: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Linear(patch_size, hidden_size)
        self.forward_calls = 0
        self.encoded_images = 0

    def reset_counters(self):
        self.forward_calls = 0
        self.encoded_images = 0

    def encode_flat(
        self,
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
    ):
        self.forward_calls += 1
        self.encoded_images += pixel_values.size(0)
        patches = self.projection(pixel_values)
        mask = pixel_attention_mask.bool()
        weights = mask.to(patches.dtype).unsqueeze(-1)
        pooled = (patches * weights).sum(1) / weights.sum(1).clamp_min(1)
        return pooled, patches, mask

    def encode_multi(
        self,
        pixel_values,
        pixel_attention_mask,
        spatial_shapes,
        image_mask,
    ):
        batch, images, patches, features = pixel_values.shape
        valid = image_mask.reshape(-1)
        pooled_out = pixel_values.new_zeros(batch * images, self.hidden_size)
        patch_out = pixel_values.new_zeros(
            batch * images,
            patches,
            self.hidden_size,
        )
        mask_out = torch.zeros(
            batch * images,
            patches,
            dtype=torch.bool,
        )
        pooled, patch_state, patch_mask = self.encode_flat(
            pixel_values.reshape(batch * images, patches, features)[valid],
            pixel_attention_mask.reshape(batch * images, patches)[valid],
            spatial_shapes.reshape(batch * images, 2)[valid],
        )
        pooled_out[valid] = pooled
        patch_out[valid] = patch_state
        mask_out[valid] = patch_mask
        return (
            pooled_out.view(batch, images, -1),
            patch_out.view(batch, images, patches, -1),
            mask_out.view(batch, images, patches),
        )


def image_inputs(batch=2, images=3, patches=2, features=6):
    generator = torch.Generator().manual_seed(7 + patches)
    pixels = torch.randn(
        batch,
        images,
        patches,
        features,
        generator=generator,
    )
    mask = torch.ones(batch, images, patches, dtype=torch.bool)
    shapes = torch.ones(batch, images, 2, dtype=torch.long)
    shapes[..., 1] = patches
    return {
        "pixel_values": pixels,
        "pixel_attention_mask": mask,
        "spatial_shapes": shapes,
    }


class SaverModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.config = SaverConfig(
            common_size=8,
            task_hidden_size=12,
            gate_hidden_size=10,
            width_embedding_size=4,
            max_span_width=4,
            top_k_images=1,
            top_r_regions=2,
            dropout=0.0,
            gate_warmup_steps=0,
        )
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 0, 0]])
        self.attention_mask = self.input_ids.ne(0)
        self.image_mask = torch.tensor([[True, True, False], [True, True, True]])
        self.global_inputs = image_inputs(patches=2)
        self.detail_inputs = image_inputs(patches=5)

    def test_mre_uses_only_selected_detail_images(self):
        vision = DummyVisionEncoder()
        model = SaverForMRE(
            self.config,
            DummyTextEncoder(),
            vision,
            num_relations=3,
        )
        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            head_spans=torch.tensor([[1, 2], [1, 2]]),
            tail_spans=torch.tensor([[3, 4], [2, 3]]),
            global_inputs=self.global_inputs,
            detail_inputs=self.detail_inputs,
            image_mask=self.image_mask,
            labels=torch.tensor([1, 2]),
            routing_mode="soft",
        )
        self.assertEqual(output.logits.shape, (2, 3))
        self.assertEqual(output.selected_indices.shape, (2, 1))
        self.assertEqual(output.selected_regions.shape, (2, 2, 2))
        self.assertTrue(
            torch.gather(
                self.image_mask,
                1,
                output.selected_indices,
            ).all()
        )
        self.assertEqual(vision.forward_calls, 2)
        self.assertLessEqual(vision.encoded_images, int(self.image_mask.sum()) + 2)
        output.loss.backward()
        gate_grad = model.router.learned_gate[-1].weight.grad
        fusion_grad = model.fusion.proposal[-1].weight.grad
        self.assertIsNotNone(gate_grad)
        self.assertIsNotNone(fusion_grad)
        self.assertGreater(float(gate_grad.abs().sum()), 0.0)
        self.assertGreater(float(fusion_grad.abs().sum()), 0.0)

    def test_text_route_skips_high_resolution_encoder(self):
        vision = DummyVisionEncoder()
        model = SaverForMRE(
            self.config,
            DummyTextEncoder(),
            vision,
            num_relations=3,
        ).eval()
        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            head_spans=torch.tensor([[1, 2], [1, 2]]),
            tail_spans=torch.tensor([[3, 4], [2, 3]]),
            global_inputs=self.global_inputs,
            detail_inputs=self.detail_inputs,
            image_mask=self.image_mask,
            routing_mode="text",
        )
        self.assertEqual(vision.forward_calls, 1)
        self.assertEqual(vision.encoded_images, int(self.image_mask.sum()))
        self.assertTrue(torch.equal(output.logits, output.text_logits))
        self.assertTrue(output.selected_regions.eq(-1).all())

    def test_span_level_mner_shapes_and_loss(self):
        vision = DummyVisionEncoder()
        model = SaverForMNER(
            self.config,
            DummyTextEncoder(),
            vision,
            num_entity_labels=4,
        )
        spans = torch.tensor(
            [
                [[1, 2], [2, 4], [3, 5]],
                [[1, 2], [2, 3], [0, 1]],
            ]
        )
        span_mask = torch.tensor([[True, True, True], [True, True, False]])
        labels = torch.tensor([[1, 0, 2], [0, 3, -100]])
        output = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            spans=spans,
            span_mask=span_mask,
            span_widths=torch.tensor([[1, 2, 2], [1, 1, 1]]),
            global_inputs=self.global_inputs,
            detail_inputs=self.detail_inputs,
            image_mask=self.image_mask,
            labels=labels,
            routing_mode="soft",
        )
        self.assertEqual(output.logits.shape, (2, 3, 4))
        self.assertEqual(output.gate_probability.shape, (2, 3))
        self.assertEqual(output.selected_indices.shape, (2, 3, 1))
        self.assertEqual(output.selected_regions.shape, (2, 3, 2, 2))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()

    def test_non_overlapping_decode(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 5.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ]
        )
        spans = torch.tensor([[[0, 2], [1, 3], [3, 4]]])
        decoded = SaverForMNER.decode_non_overlapping(
            logits,
            spans,
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(
            [(item["start"], item["end"]) for item in decoded[0]],
            [(0, 2), (3, 4)],
        )


class CalibrationTest(unittest.TestCase):
    def test_feasible_and_no_feasible_fallback(self):
        calibrator = RiskControlledCalibrator(
            target_risk=0.20,
            confidence=0.80,
            grid_size=5,
        )
        scores = torch.linspace(0, 1, 100)
        feasible = calibrator.calibrate(
            scores,
            torch.zeros(100, dtype=torch.bool),
        )
        self.assertEqual(feasible["threshold"], 0.0)
        self.assertEqual(feasible["coverage"], 1.0)
        self.assertEqual(feasible["grid_size"], 5.0)
        fallback = calibrator.calibrate(
            scores,
            torch.ones(100, dtype=torch.bool),
        )
        self.assertGreater(fallback["threshold"], 1.0)
        self.assertEqual(fallback["coverage"], 0.0)


class AttachmentSelectionTest(unittest.TestCase):
    def test_facility_requires_a_real_selected_set(self):
        with self.assertRaisesRegex(ValueError, "top_k_images>=2"):
            SaverConfig(
                top_k_images=1,
                selection_type="facility",
            ).validate()

    def test_facility_prefers_complementary_evidence(self):
        router = SaverRouter(hidden_size=2, gate_hidden_size=4)
        query = torch.tensor([[1.0, 0.0]])
        candidates = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.999, 0.045],
                    [0.60, 0.80],
                ]
            ]
        )
        image_mask = torch.ones(1, 3, dtype=torch.bool)

        topk, _, _ = router.select(
            query,
            candidates,
            image_mask,
            top_k=2,
            selection_type="topk",
        )
        facility, weights, _ = router.select(
            query,
            candidates,
            image_mask,
            top_k=2,
            selection_type="facility",
            facility_lambda_relevance=0.25,
            facility_lambda_coverage=2.0,
        )

        self.assertEqual(set(topk[0].tolist()), {0, 1})
        self.assertIn(2, facility[0].tolist())
        self.assertEqual(
            len(set(facility[0].tolist()).intersection({0, 1})),
            1,
        )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

        sparse_mask = torch.tensor([[True, False, False]])
        sparse_indices, sparse_weights, _ = router.select(
            query,
            candidates,
            sparse_mask,
            top_k=2,
            selection_type="facility",
        )
        selected_valid = torch.gather(sparse_mask, 1, sparse_indices)
        self.assertEqual(int(selected_valid.sum()), 1)
        self.assertAlmostEqual(float(sparse_weights.sum()), 1.0, places=6)

        one_candidate, _, _ = router.select(
            query,
            candidates[:, :1],
            torch.ones(1, 1, dtype=torch.bool),
            top_k=2,
            selection_type="facility",
        )
        self.assertEqual(one_candidate.tolist(), [[0]])


if __name__ == "__main__":
    unittest.main()
