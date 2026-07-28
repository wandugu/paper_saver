"""SAVER: Selective Adaptive Visual Evidence Routing.

This module implements the architecture described by the SAVER manuscript:

* ModernBERT supplies token representations.
* SigLIP-2 NaFlex screens every attached image at a small patch budget.
* A unit-level gate decides whether a span/entity pair needs detailed vision.
* Top-1 attachment retrieval is the default and only selected attachments are
  re-encoded at the larger detail patch budget.
* An optional relevance-weighted facility-location selector is available only
  for real K>=2 sensitivity experiments.
* Query-conditioned patch pooling supplies evidence to MRE or span-level MNER.

The task heads accept injected text/vision encoders.  Unit tests can therefore
exercise routing and conditional computation without downloading checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


TensorDict = Mapping[str, torch.Tensor]


@dataclass
class SaverConfig:
    """Configuration shared by the MRE and MNER implementations."""

    text_model_name: str = "answerdotai/ModernBERT-base"
    vision_model_name: str = "google/siglip2-base-patch16-naflex"
    common_size: int = 256
    task_hidden_size: int = 384
    gate_hidden_size: int = 256
    width_embedding_size: int = 32
    max_span_width: int = 10
    top_k_images: int = 1
    selection_type: str = "topk"
    facility_lambda_relevance: float = 1.0
    facility_lambda_coverage: float = 1.0
    top_r_regions: int = 3
    screen_max_num_patches: int = 64
    detail_max_num_patches: int = 256
    gate_threshold: float = 0.5
    target_activation: float = 0.35
    utility_margin: float = 0.0
    gate_warmup_steps: int = 500
    lambda_utility: float = 0.25
    lambda_budget: float = 0.05
    lambda_consistency: float = 0.05
    dropout: float = 0.1
    freeze_text_encoder: bool = False
    freeze_vision_encoder: bool = False
    attn_implementation: str = "sdpa"

    def validate(self) -> None:
        if self.top_k_images < 1:
            raise ValueError("top_k_images must be positive")
        if self.selection_type not in {"topk", "facility"}:
            raise ValueError("selection_type must be 'topk' or 'facility'")
        if self.selection_type == "facility" and self.top_k_images < 2:
            raise ValueError(
                "facility selection requires top_k_images>=2; K=1 does not "
                "support a cross-selected-image diversity claim"
            )
        if self.facility_lambda_relevance < 0:
            raise ValueError("facility_lambda_relevance must be nonnegative")
        if self.facility_lambda_coverage < 0:
            raise ValueError("facility_lambda_coverage must be nonnegative")
        if (
            self.selection_type == "facility"
            and self.facility_lambda_relevance == 0
            and self.facility_lambda_coverage == 0
        ):
            raise ValueError("facility selection requires a nonzero objective")
        if self.top_r_regions < 1:
            raise ValueError("top_r_regions must be positive")
        if self.max_span_width < 1:
            raise ValueError("max_span_width must be positive")
        if self.common_size < 1 or self.task_hidden_size < 1:
            raise ValueError("hidden sizes must be positive")
        if self.common_size % 4:
            raise ValueError("common_size must be divisible by four")
        if self.screen_max_num_patches < 1:
            raise ValueError("screen_max_num_patches must be positive")
        if self.detail_max_num_patches <= self.screen_max_num_patches:
            raise ValueError(
                "detail_max_num_patches must exceed screen_max_num_patches"
            )
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be in [0, 1]")
        if not 0.0 <= self.target_activation <= 1.0:
            raise ValueError("target_activation must be in [0, 1]")
        if self.utility_margin < 0:
            raise ValueError("utility_margin must be nonnegative")
        if self.gate_warmup_steps < 0:
            raise ValueError("gate_warmup_steps must be nonnegative")
        if any(
            value < 0
            for value in (
                self.lambda_utility,
                self.lambda_budget,
                self.lambda_consistency,
            )
        ):
            raise ValueError("loss weights must be nonnegative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SaverConfig":
        known = {item.name for item in field_list(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("unknown SaverConfig keys: {}".format(", ".join(unknown)))
        config = cls(**dict(values))
        config.validate()
        return config


def field_list(config_type: type) -> Sequence[Any]:
    """Small compatibility wrapper around dataclasses.fields."""

    from dataclasses import fields

    return fields(config_type)


@dataclass
class SaverOutput:
    """Stable output contract shared by the MRE and MNER models."""

    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    text_logits: torch.Tensor
    visual_logits: torch.Tensor
    gate_probability: torch.Tensor
    gate_active: torch.Tensor
    selected_indices: torch.Tensor
    attachment_relevance: torch.Tensor
    selected_regions: torch.Tensor
    utility_target: Optional[torch.Tensor] = None
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)


class ModernBertTextEncoder(nn.Module):
    """Thin adapter around a Hugging Face ModernBERT checkpoint."""

    def __init__(self, backbone: nn.Module, hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(
            hidden_size
            if hidden_size is not None
            else getattr(getattr(backbone, "config", None), "hidden_size")
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "answerdotai/ModernBERT-base",
        freeze: bool = False,
        attn_implementation: str = "sdpa",
        **kwargs: Any,
    ) -> "ModernBertTextEncoder":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "SAVER requires transformers>=4.49 for ModernBERT and SigLIP-2"
            ) from exc
        load_kwargs = dict(kwargs)
        if attn_implementation:
            load_kwargs.setdefault("attn_implementation", attn_implementation)
        backbone = AutoModel.from_pretrained(model_name, **load_kwargs)
        encoder = cls(backbone)
        if freeze:
            encoder.requires_grad_(False)
        return encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        return outputs.last_hidden_state


class Siglip2VisionEncoder(nn.Module):
    """Adapter exposing pooled and patch-level SigLIP-2 NaFlex features."""

    def __init__(self, backbone: nn.Module, hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        self.hidden_size = int(
            hidden_size if hidden_size is not None else getattr(config, "hidden_size")
        )
        self.forward_calls = 0
        self.encoded_images = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "google/siglip2-base-patch16-naflex",
        freeze: bool = False,
        attn_implementation: str = "sdpa",
        **kwargs: Any,
    ) -> "Siglip2VisionEncoder":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "SAVER requires transformers>=4.49 for ModernBERT and SigLIP-2"
            ) from exc
        load_kwargs = dict(kwargs)
        if attn_implementation:
            load_kwargs.setdefault("attn_implementation", attn_implementation)
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        backbone = getattr(model, "vision_model", model)
        encoder = cls(backbone)
        if freeze:
            encoder.requires_grad_(False)
        return encoder

    def reset_counters(self) -> None:
        self.forward_calls = 0
        self.encoded_images = 0

    def encode_flat(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a flat image batch.

        NaFlex ``pixel_values`` has shape ``[images, patches, patch_features]``.
        The returned tuple is ``(pooled, patches, valid_patch_mask)``.
        """

        if pixel_values.dim() != 3:
            raise ValueError(
                "NaFlex pixel_values must have shape [images, patches, patch_features]"
            )
        if pixel_attention_mask.shape != pixel_values.shape[:2]:
            raise ValueError("pixel_attention_mask does not match pixel_values")
        if spatial_shapes.shape != (pixel_values.size(0), 2):
            raise ValueError("spatial_shapes must have shape [images, 2]")

        self.forward_calls += 1
        self.encoded_images += int(pixel_values.size(0))
        outputs = self.backbone(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            return_dict=True,
        )
        patches = outputs.last_hidden_state
        patch_mask = pixel_attention_mask.bool()
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            weights = patch_mask.to(patches.dtype).unsqueeze(-1)
            pooled = (patches * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return pooled, patches, patch_mask

    def encode_multi(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode valid members of a padded ``[batch, images, ...]`` batch."""

        if pixel_values.dim() != 4:
            raise ValueError(
                "multi-image pixel_values must have shape "
                "[batch, images, patches, patch_features]"
            )
        batch_size, image_count, patch_count = pixel_values.shape[:3]
        if image_mask.shape != (batch_size, image_count):
            raise ValueError("image_mask shape does not match the image batch")
        if pixel_attention_mask.shape != (batch_size, image_count, patch_count):
            raise ValueError("multi-image pixel_attention_mask has invalid shape")
        if spatial_shapes.shape != (batch_size, image_count, 2):
            raise ValueError("multi-image spatial_shapes has invalid shape")

        valid = image_mask.bool().reshape(-1)
        flat_pixels = pixel_values.reshape(
            batch_size * image_count,
            patch_count,
            pixel_values.size(-1),
        )
        flat_patch_mask = pixel_attention_mask.reshape(
            batch_size * image_count,
            patch_count,
        )
        flat_shapes = spatial_shapes.reshape(batch_size * image_count, 2)

        pooled = pixel_values.new_zeros(
            batch_size * image_count,
            self.hidden_size,
        )
        patches = pixel_values.new_zeros(
            batch_size * image_count,
            patch_count,
            self.hidden_size,
        )
        returned_mask = torch.zeros(
            batch_size * image_count,
            patch_count,
            dtype=torch.bool,
            device=pixel_values.device,
        )
        if valid.any():
            valid_pooled, valid_patches, valid_mask = self.encode_flat(
                flat_pixels[valid],
                flat_patch_mask[valid],
                flat_shapes[valid],
            )
            pooled[valid] = valid_pooled
            patches[valid] = valid_patches
            returned_mask[valid] = valid_mask
        return (
            pooled.view(batch_size, image_count, -1),
            patches.view(batch_size, image_count, patch_count, -1),
            returned_mask.view(batch_size, image_count, patch_count),
        )


class SaverRouter(nn.Module):
    """Unit-level gate plus image relevance scoring.

    ``learned`` and ``maxsim`` have identical access to global SigLIP-2
    features, enabling a fair parameterized-vs-parameter-free comparison.
    ``entropy``, ``margin``, and ``text`` are explicitly text-only controls.
    """

    SUPPORTED_GATE_TYPES = {
        "learned",
        "maxsim",
        "entropy",
        "margin",
        "always",
        "never",
        "text",
    }

    def __init__(
        self,
        hidden_size: int,
        gate_hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.learned_gate = nn.Sequential(
            nn.Linear(hidden_size * 4 + 4, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 1),
        )
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 1),
        )

    @staticmethod
    def _as_units(query: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if query.dim() == 2:
            return query.unsqueeze(1), True
        if query.dim() == 3:
            return query, False
        raise ValueError(
            "query must have shape [batch, hidden] or [batch, units, hidden]"
        )

    @staticmethod
    def _text_probabilities(
        text_logits: Optional[torch.Tensor],
        gate_type: str,
    ) -> torch.Tensor:
        if text_logits is None:
            raise ValueError("{} gate requires text_logits".format(gate_type))
        logits = text_logits if text_logits.dim() == 3 else text_logits.unsqueeze(1)
        probabilities = F.softmax(logits, dim=-1)
        if gate_type == "entropy":
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
            return (entropy / math.log(max(2, probabilities.size(-1)))).clamp(0, 1)
        top_two = probabilities.topk(k=min(2, probabilities.size(-1)), dim=-1).values
        if top_two.size(-1) == 1:
            margin = top_two[..., 0]
        else:
            margin = top_two[..., 0] - top_two[..., 1]
        return (1.0 - margin).clamp(0, 1)

    def score(
        self,
        query: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        gate_type: str = "learned",
        text_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gate_type not in self.SUPPORTED_GATE_TYPES:
            raise ValueError(
                "unsupported gate_type={!r}; choose one of {}".format(
                    gate_type,
                    sorted(self.SUPPORTED_GATE_TYPES),
                )
            )
        unit_query, squeezed = self._as_units(query)
        unit_query = F.normalize(unit_query, dim=-1)
        candidates = F.normalize(candidate_embeddings, dim=-1)
        similarities = torch.einsum("buh,bmh->bum", unit_query, candidates)
        mask = candidate_mask.bool().unsqueeze(1)
        masked_similarities = similarities.masked_fill(~mask, -1e4)
        max_similarity, max_index = masked_similarities.max(dim=-1)
        has_image = candidate_mask.bool().any(dim=-1, keepdim=True)

        valid_count = mask.sum(-1).clamp_min(1)
        safe_similarities = similarities.masked_fill(~mask, 0.0)
        mean_similarity = safe_similarities.sum(-1) / valid_count
        variance = ((similarities - mean_similarity.unsqueeze(-1)) ** 2).masked_fill(
            ~mask, 0.0
        ).sum(-1) / valid_count
        std_similarity = variance.sqrt()
        top_count = min(2, candidate_embeddings.size(1))
        top_values = masked_similarities.topk(k=top_count, dim=-1).values
        top_valid = (
            candidate_mask.sum(-1, keepdim=True)
            .clamp(max=top_count)
            .to(top_values.dtype)
        )
        top_values = top_values.masked_fill(top_values < -1e3, 0.0)
        top_two_mean = top_values.sum(-1) / top_valid.clamp_min(1.0)

        if gate_type == "learned":
            expanded = candidates.unsqueeze(1).expand(
                -1,
                unit_query.size(1),
                -1,
                -1,
            )
            gather_index = (
                max_index.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(
                    -1,
                    -1,
                    1,
                    candidates.size(-1),
                )
            )
            best_visual = torch.gather(expanded, 2, gather_index).squeeze(2)
            features = torch.cat(
                [
                    unit_query,
                    best_visual,
                    unit_query * best_visual,
                    torch.abs(unit_query - best_visual),
                    max_similarity.unsqueeze(-1),
                    mean_similarity.unsqueeze(-1),
                    std_similarity.unsqueeze(-1),
                    top_two_mean.unsqueeze(-1),
                ],
                dim=-1,
            )
            probability = torch.sigmoid(self.learned_gate(features).squeeze(-1))
        elif gate_type == "maxsim":
            probability = ((max_similarity + 1.0) / 2.0).clamp(0, 1)
        elif gate_type in {"entropy", "margin"}:
            probability = self._text_probabilities(text_logits, gate_type)
        elif gate_type == "text":
            probability = torch.sigmoid(self.text_gate(unit_query).squeeze(-1))
        elif gate_type == "always":
            probability = torch.ones_like(max_similarity)
        else:
            probability = torch.zeros_like(max_similarity)

        probability = probability * has_image.to(probability.dtype)
        if squeezed:
            probability = probability.squeeze(1)
            masked_similarities = masked_similarities.squeeze(1)
        return probability, masked_similarities

    def select(
        self,
        query: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        top_k: int = 1,
        selection_type: str = "topk",
        facility_lambda_relevance: float = 1.0,
        facility_lambda_coverage: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return indices, normalized selected weights, and all relevance."""

        unit_query, squeezed = self._as_units(query)
        query_normalized = F.normalize(unit_query, dim=-1)
        candidates = F.normalize(candidate_embeddings, dim=-1)
        relevance = torch.einsum("buh,bmh->bum", query_normalized, candidates)
        relevance = relevance.masked_fill(
            ~candidate_mask.bool().unsqueeze(1),
            -1e4,
        )
        requested_k = max(1, top_k)
        k = min(requested_k, relevance.size(-1))
        if selection_type == "topk":
            selected_relevance, selected_indices = relevance.topk(k=k, dim=-1)
        elif selection_type == "facility":
            if requested_k < 2:
                raise ValueError(
                    "facility selection requires top_k>=2; use topk for K=1"
                )
            if k == 1:
                # A batch can contain only one attachment even when the
                # experiment requests K>=2.  In that degenerate sample there
                # is no set choice to make, so use the sole valid candidate.
                selected_relevance, selected_indices = relevance.topk(
                    k=1,
                    dim=-1,
                )
            else:
                selected_indices = self._facility_location_select(
                    relevance,
                    candidates,
                    candidate_mask,
                    k,
                    facility_lambda_relevance,
                    facility_lambda_coverage,
                )
                selected_relevance = torch.gather(
                    relevance,
                    2,
                    selected_indices,
                )
        else:
            raise ValueError("selection_type must be 'topk' or 'facility'")
        expanded_mask = (
            candidate_mask.bool()
            .unsqueeze(1)
            .expand(
                -1,
                unit_query.size(1),
                -1,
            )
        )
        selected_valid = torch.gather(expanded_mask, 2, selected_indices)
        weights = F.softmax(selected_relevance, dim=-1)
        weights = weights * selected_valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        if squeezed:
            return (
                selected_indices.squeeze(1),
                weights.squeeze(1),
                relevance.squeeze(1),
            )
        return selected_indices, weights, relevance

    @staticmethod
    def _facility_location_select(
        relevance: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor,
        top_k: int,
        lambda_relevance: float,
        lambda_coverage: float,
    ) -> torch.Tensor:
        """Greedy relevance-weighted facility-location selection for K>1."""

        batch_size, unit_count, image_count = relevance.shape
        relevance_01 = ((relevance + 1.0) / 2.0).clamp(0.0, 1.0)
        valid_images = candidate_mask.bool()
        relevance_01 = relevance_01.masked_fill(
            ~valid_images.unsqueeze(1),
            0.0,
        )
        pairwise = torch.einsum(
            "bih,bjh->bij",
            candidate_embeddings,
            candidate_embeddings,
        )
        pairwise = ((pairwise + 1.0) / 2.0).clamp(0.0, 1.0)
        pairwise = pairwise.masked_fill(
            ~valid_images.unsqueeze(1),
            0.0,
        )
        pairwise = pairwise.masked_fill(
            ~valid_images.unsqueeze(2),
            0.0,
        )

        current_coverage = relevance.new_zeros(
            batch_size,
            unit_count,
            image_count,
        )
        already_selected = torch.zeros(
            batch_size,
            unit_count,
            image_count,
            dtype=torch.bool,
            device=relevance.device,
        )
        selected: List[torch.Tensor] = []
        expanded_pairwise = pairwise.unsqueeze(1)
        valid_candidates = valid_images.unsqueeze(1).expand(
            -1,
            unit_count,
            -1,
        )
        for _ in range(top_k):
            improvement = (expanded_pairwise - current_coverage.unsqueeze(2)).clamp_min(
                0.0
            )
            coverage_gain = (improvement * relevance_01.unsqueeze(2)).sum(-1)
            marginal = (
                float(lambda_relevance) * relevance_01
                + float(lambda_coverage) * coverage_gain
            )
            eligible = valid_candidates & ~already_selected
            marginal = marginal.masked_fill(~eligible, -1e4)
            has_eligible = eligible.any(-1)
            best_index = marginal.argmax(-1)
            # A fixed output width is convenient for batching.  If a sample
            # contains fewer than K valid attachments, fill the remaining
            # slots with a masked attachment instead of duplicating evidence.
            masked_fallback = (~valid_candidates).to(torch.long).argmax(-1)
            next_index = torch.where(
                has_eligible,
                best_index,
                masked_fallback,
            )
            selected.append(next_index)
            already_selected.scatter_(
                2,
                next_index.unsqueeze(-1),
                True,
            )
            gather_index = (
                next_index.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(
                    -1,
                    -1,
                    1,
                    image_count,
                )
            )
            selected_coverage = torch.gather(
                expanded_pairwise.expand(-1, unit_count, -1, -1),
                2,
                gather_index,
            ).squeeze(2)
            current_coverage = torch.maximum(
                current_coverage,
                selected_coverage,
            )
        return torch.stack(selected, dim=-1)


def gather_image_values(
    values: torch.Tensor,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[batch, images, ...]`` values for ``[batch, units, K]`` indices."""

    if selected_indices.dim() == 2:
        selected_indices = selected_indices.unsqueeze(1)
        squeeze = True
    elif selected_indices.dim() == 3:
        squeeze = False
    else:
        raise ValueError("selected_indices must have shape [B,K] or [B,U,K]")
    batch_size, unit_count, selected_count = selected_indices.shape
    if values.size(0) != batch_size:
        raise ValueError("batch dimension mismatch")
    batch_index = torch.arange(batch_size, device=values.device)
    batch_index = batch_index.view(batch_size, 1, 1).expand(
        -1,
        unit_count,
        selected_count,
    )
    result = values[batch_index, selected_indices]
    return result.squeeze(1) if squeeze else result


class QueryEvidencePooler(nn.Module):
    """Pool selected global and patch evidence with a unit query."""

    def __init__(
        self,
        hidden_size: int,
        top_r_regions: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.top_r_regions = top_r_regions
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        query: torch.Tensor,
        selected_global: torch.Tensor,
        detail_patches: torch.Tensor,
        detail_patch_mask: torch.Tensor,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.dim() != 3:
            raise ValueError("query must have shape [batch, units, hidden]")
        batch_size, unit_count, hidden_size = query.shape
        if hidden_size != self.hidden_size:
            raise ValueError("query hidden size mismatch")
        if selected_global.dim() != 4 or detail_patches.dim() != 4:
            raise ValueError("selected evidence tensors have invalid rank")
        if selected_indices.shape != selected_valid.shape:
            raise ValueError("selected indices and validity masks must match")

        selected_count = selected_global.size(2)
        image_count = detail_patches.size(1)
        patch_count = detail_patches.size(2)
        if detail_patch_mask.shape != (
            batch_size,
            image_count,
            patch_count,
        ):
            raise ValueError("detail patch mask has invalid shape")
        if selected_indices.shape != (
            batch_size,
            unit_count,
            selected_count,
        ):
            raise ValueError("selected attachment indices have invalid shape")

        # Compute unit-to-patch similarities without expanding every patch
        # representation over every MNER span.  Only the compact similarity
        # tensor has a unit dimension; feature gathering happens after Top-R.
        all_region_similarity = torch.einsum(
            "buh,bmph->bump",
            F.normalize(query, dim=-1),
            F.normalize(detail_patches, dim=-1),
        )
        gather_images = selected_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            patch_count,
        )
        selected_similarity = torch.gather(
            all_region_similarity,
            2,
            gather_images,
        )
        expanded_patch_mask = detail_patch_mask.unsqueeze(1).expand(
            -1,
            unit_count,
            -1,
            -1,
        )
        selected_patch_mask = torch.gather(
            expanded_patch_mask,
            2,
            gather_images,
        )
        selected_patch_mask = selected_patch_mask & selected_valid.unsqueeze(-1)
        region_similarity = selected_similarity.reshape(
            batch_size,
            unit_count,
            selected_count * patch_count,
        )
        flat_patch_mask = selected_patch_mask.reshape_as(region_similarity)
        region_similarity = region_similarity.masked_fill(~flat_patch_mask, -1e4)
        region_count = min(self.top_r_regions, region_similarity.size(2))
        region_scores, region_indices = region_similarity.topk(
            k=region_count,
            dim=-1,
        )
        selection_slots = torch.div(
            region_indices,
            patch_count,
            rounding_mode="floor",
        )
        patch_indices = region_indices.remainder(patch_count)
        image_indices = torch.gather(
            selected_indices,
            2,
            selection_slots,
        )
        batch_indices = (
            torch.arange(
                batch_size,
                device=query.device,
            )
            .view(batch_size, 1, 1)
            .expand_as(image_indices)
        )
        top_regions = detail_patches[
            batch_indices,
            image_indices,
            patch_indices,
        ]
        top_region_mask = region_scores > -1e3
        selected_regions = torch.stack(
            [image_indices, patch_indices],
            dim=-1,
        )
        selected_regions = selected_regions.masked_fill(
            ~top_region_mask.unsqueeze(-1),
            -1,
        )
        if top_regions.shape != (
            batch_size,
            unit_count,
            region_count,
            hidden_size,
        ):
            raise RuntimeError("Top-R region gathering produced an invalid shape")

        evidence_tokens = torch.cat([selected_global, top_regions], dim=2)
        evidence_mask = torch.cat([selected_valid, top_region_mask], dim=2)
        flat_query = query.reshape(batch_size * unit_count, 1, hidden_size)
        flat_tokens = evidence_tokens.reshape(
            batch_size * unit_count,
            evidence_tokens.size(2),
            hidden_size,
        )
        flat_mask = evidence_mask.reshape(
            batch_size * unit_count,
            evidence_mask.size(2),
        )
        has_evidence = flat_mask.any(-1)
        safe_tokens = flat_tokens.clone()
        safe_mask = flat_mask.clone()
        if (~has_evidence).any():
            safe_tokens[~has_evidence, 0] = 0
            safe_mask[~has_evidence, 0] = True
        attended, _ = self.attention(
            flat_query,
            safe_tokens,
            safe_tokens,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        pooled = self.norm(flat_query + attended)
        pooled = self.norm(pooled + self.ffn(pooled))
        pooled = pooled.squeeze(1)
        pooled = pooled * has_evidence.to(pooled.dtype).unsqueeze(-1)
        pooled = pooled.view(batch_size, unit_count, hidden_size)
        return (
            pooled,
            has_evidence.view(batch_size, unit_count),
            selected_regions,
        )


class SpanRepresentation(nn.Module):
    """Start/end/max-pool/width representation for span-level MNER."""

    def __init__(
        self,
        token_hidden_size: int,
        max_span_width: int,
        width_embedding_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_span_width = max_span_width
        self.width_embedding = nn.Embedding(
            max_span_width + 1,
            width_embedding_size,
        )
        raw_size = token_hidden_size * 3 + width_embedding_size
        self.projection = nn.Sequential(
            nn.Linear(raw_size, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_states: torch.Tensor,
        spans: torch.Tensor,
        span_widths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if spans.dim() != 3 or spans.size(-1) != 2:
            raise ValueError("spans must have shape [batch, spans, 2]")
        batch_size, span_count = spans.shape[:2]
        sequence_length = token_states.size(1)
        starts = spans[..., 0].clamp(0, sequence_length - 1)
        ends = spans[..., 1].clamp(1, sequence_length)
        end_tokens = (ends - 1).clamp(0, sequence_length - 1)
        batch_index = torch.arange(batch_size, device=token_states.device)
        batch_index = batch_index.view(batch_size, 1).expand(-1, span_count)
        start_state = token_states[batch_index, starts]
        end_state = token_states[batch_index, end_tokens]

        token_widths = (ends - starts).clamp_min(1)
        maximum_token_width = int(token_widths.detach().max().item())
        offsets = torch.arange(
            maximum_token_width,
            device=token_states.device,
        )
        token_positions = (starts.unsqueeze(-1) + offsets.view(1, 1, -1)).clamp_max(
            sequence_length - 1
        )
        expanded_batch = batch_index.unsqueeze(-1).expand_as(token_positions)
        span_states = token_states[
            expanded_batch,
            token_positions,
        ]
        in_span = offsets.view(1, 1, -1) < token_widths.unsqueeze(-1)
        pooled = span_states.masked_fill(
            ~in_span.unsqueeze(-1),
            torch.finfo(token_states.dtype).min,
        )
        pooled = pooled.max(dim=2).values
        invalid = ends <= starts
        pooled = pooled.masked_fill(invalid.unsqueeze(-1), 0.0)

        if span_widths is None:
            span_widths = (ends - starts).clamp(1, self.max_span_width)
        width = span_widths.clamp(1, self.max_span_width)
        width_state = self.width_embedding(width)
        raw = torch.cat([start_state, end_state, pooled, width_state], dim=-1)
        return self.projection(raw)


class EntityPairRepresentation(nn.Module):
    """Build a direct marked-pair query for relation extraction."""

    def __init__(
        self,
        token_hidden_size: int,
        max_span_width: int,
        width_embedding_size: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.entity_encoder = SpanRepresentation(
            token_hidden_size,
            max_span_width,
            width_embedding_size,
            output_size,
            dropout,
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(output_size * 4, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_states: torch.Tensor,
        head_spans: torch.Tensor,
        tail_spans: torch.Tensor,
    ) -> torch.Tensor:
        if head_spans.dim() == 2:
            head_spans = head_spans.unsqueeze(1)
        if tail_spans.dim() == 2:
            tail_spans = tail_spans.unsqueeze(1)
        head = self.entity_encoder(token_states, head_spans).squeeze(1)
        tail = self.entity_encoder(token_states, tail_spans).squeeze(1)
        return self.pair_projection(
            torch.cat([head, tail, head * tail, torch.abs(head - tail)], dim=-1)
        )


class GatedFusion(nn.Module):
    """Residual multimodal fusion that is exactly text-only at gate value zero."""

    def __init__(
        self,
        task_size: int,
        evidence_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.evidence_projection = nn.Linear(evidence_size, task_size)
        self.proposal = nn.Sequential(
            nn.Linear(task_size * 4, task_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(task_size * 2, task_size),
        )
        self.norm = nn.LayerNorm(task_size)

    def forward(
        self,
        text_state: torch.Tensor,
        evidence: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.evidence_projection(evidence)
        features = torch.cat(
            [
                text_state,
                visual,
                text_state * visual,
                torch.abs(text_state - visual),
            ],
            dim=-1,
        )
        proposed = self.norm(self.proposal(features))
        return text_state + gate_weight.unsqueeze(-1) * (proposed - text_state)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def saver_auxiliary_losses(
    gate_probability: torch.Tensor,
    text_unit_loss: torch.Tensor,
    visual_unit_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    target_activation: float,
    utility_margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visual-utility BCE, activation-budget loss, and utility labels."""

    utility_target = (
        visual_unit_loss.detach() + float(utility_margin) < text_unit_loss.detach()
    ).to(gate_probability.dtype)
    valid = valid_mask.bool()
    if not valid.any():
        zero = gate_probability.sum() * 0.0
        return zero, zero, utility_target
    probability = gate_probability[valid].clamp(1e-6, 1.0 - 1e-6)
    target = utility_target[valid]
    utility_loss = F.binary_cross_entropy(probability, target)
    activation = probability.mean()
    budget_loss = (activation - float(target_activation)) ** 2
    return utility_loss, budget_loss, utility_target


class _SaverTaskBase(nn.Module):
    """Shared global-screening, conditional-detail, and fusion path."""

    def __init__(
        self,
        config: SaverConfig,
        text_encoder: nn.Module,
        vision_encoder: Siglip2VisionEncoder,
        text_hidden_size: int,
        vision_hidden_size: int,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        # Routing queries are task-unit states (pair/span), not raw token
        # states, so this projection starts at task_hidden_size.
        self.text_to_common = nn.Linear(
            config.task_hidden_size,
            config.common_size,
        )
        self.vision_to_common = nn.Linear(vision_hidden_size, config.common_size)
        self.router = SaverRouter(
            config.common_size,
            config.gate_hidden_size,
            dropout=config.dropout,
        )
        self.evidence_pooler = QueryEvidencePooler(
            config.common_size,
            config.top_r_regions,
            dropout=config.dropout,
        )
        self.fusion = GatedFusion(
            config.task_hidden_size,
            config.common_size,
            config.dropout,
        )

    @staticmethod
    def _routing_masks(
        probability: torch.Tensor,
        unit_mask: torch.Tensor,
        routing_mode: str,
        threshold: float,
        gate_floor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if routing_mode not in {"soft", "hard", "vision", "text"}:
            raise ValueError("routing_mode must be soft, hard, vision, or text")
        valid = unit_mask.bool()
        if routing_mode == "text":
            active = torch.zeros_like(valid)
            mix = torch.zeros_like(probability)
        elif routing_mode == "vision":
            active = valid
            mix = valid.to(probability.dtype)
        elif routing_mode == "soft":
            active = valid
            floor = min(max(float(gate_floor), 0.0), 1.0)
            mix = floor + (1.0 - floor) * probability
            mix = mix * valid.to(probability.dtype)
        else:
            active = (probability >= float(threshold)) & valid
            mix = active.to(probability.dtype)
        return active, mix

    def _encode_selected_detail(
        self,
        selected_indices: torch.Tensor,
        active_units: torch.Tensor,
        detail_inputs: TensorDict,
        image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode each distinct selected attachment at most once."""

        if selected_indices.dim() != 3:
            raise ValueError("selected_indices must have shape [B,U,K]")
        pixels = detail_inputs["pixel_values"]
        patch_mask = detail_inputs["pixel_attention_mask"]
        spatial_shapes = detail_inputs["spatial_shapes"]
        batch_size, image_count, patch_count = pixels.shape[:3]

        selected_valid = gather_image_values(
            image_mask.bool(),
            selected_indices,
        )
        selected_valid = selected_valid & active_units.unsqueeze(-1)
        output_patches = pixels.new_zeros(
            batch_size,
            image_count,
            patch_count,
            self.vision_encoder.hidden_size,
        )
        output_mask = torch.zeros(
            batch_size,
            image_count,
            patch_count,
            dtype=torch.bool,
            device=pixels.device,
        )
        if not selected_valid.any():
            return output_patches, output_mask, selected_valid

        batch_index = torch.arange(batch_size, device=pixels.device)
        batch_index = batch_index.view(batch_size, 1, 1).expand_as(selected_indices)
        keys = batch_index * image_count + selected_indices
        flat_valid = selected_valid.reshape(-1)
        valid_positions = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        unique_keys = torch.unique(
            keys.reshape(-1)[valid_positions],
            sorted=True,
        )
        flat_pixels = pixels.reshape(
            batch_size * image_count,
            patch_count,
            pixels.size(-1),
        )
        flat_patch_mask = patch_mask.reshape(
            batch_size * image_count,
            patch_count,
        )
        flat_shapes = spatial_shapes.reshape(batch_size * image_count, 2)
        _, patches, returned_mask = self.vision_encoder.encode_flat(
            flat_pixels[unique_keys],
            flat_patch_mask[unique_keys],
            flat_shapes[unique_keys],
        )
        out_patch_flat = output_patches.reshape(
            batch_size * image_count,
            patch_count,
            -1,
        )
        out_mask_flat = output_mask.reshape(
            batch_size * image_count,
            patch_count,
        )
        out_patch_flat[unique_keys] = patches
        out_mask_flat[unique_keys] = returned_mask
        return output_patches, output_mask, selected_valid

    def _route_and_pool(
        self,
        unit_query: torch.Tensor,
        text_logits: torch.Tensor,
        unit_mask: torch.Tensor,
        global_inputs: TensorDict,
        detail_inputs: TensorDict,
        image_mask: torch.Tensor,
        gate_type: str,
        routing_mode: str,
        threshold: Optional[float],
        gate_floor: float,
    ) -> Dict[str, torch.Tensor]:
        pooled, _, _ = self.vision_encoder.encode_multi(
            global_inputs["pixel_values"],
            global_inputs["pixel_attention_mask"],
            global_inputs["spatial_shapes"],
            image_mask,
        )
        global_common = self.vision_to_common(pooled)
        global_common = F.normalize(global_common, dim=-1)
        query_common = F.normalize(self.text_to_common(unit_query), dim=-1)
        gate_probability, _ = self.router.score(
            query_common,
            global_common,
            image_mask,
            gate_type=gate_type,
            text_logits=text_logits,
        )
        gate_probability = gate_probability * unit_mask.to(gate_probability.dtype)
        selected_indices, selected_weights, relevance = self.router.select(
            query_common,
            global_common,
            image_mask,
            top_k=self.config.top_k_images,
            selection_type=self.config.selection_type,
            facility_lambda_relevance=self.config.facility_lambda_relevance,
            facility_lambda_coverage=self.config.facility_lambda_coverage,
        )
        active, mix = self._routing_masks(
            gate_probability,
            unit_mask,
            routing_mode,
            self.config.gate_threshold if threshold is None else threshold,
            gate_floor,
        )
        detail_patches, detail_patch_mask, selected_valid = (
            self._encode_selected_detail(
                selected_indices,
                active,
                detail_inputs,
                image_mask,
            )
        )
        selected_global = gather_image_values(global_common, selected_indices)
        selected_global = selected_global * selected_weights.unsqueeze(-1)
        detail_common = self.vision_to_common(detail_patches)
        evidence, has_evidence, selected_regions = self.evidence_pooler(
            query_common,
            selected_global,
            detail_common,
            detail_patch_mask,
            selected_indices,
            selected_valid,
        )
        consistency = 1.0 - F.cosine_similarity(
            query_common,
            evidence,
            dim=-1,
            eps=1e-8,
        )
        consistency = consistency * has_evidence.to(consistency.dtype)
        return {
            "evidence": evidence,
            "gate_probability": gate_probability,
            "gate_active": active,
            "mix": mix,
            "selected_indices": selected_indices,
            "attachment_relevance": relevance,
            "selected_regions": selected_regions,
            "consistency": consistency,
            "has_evidence": has_evidence,
        }

    def _combine_losses(
        self,
        task_loss: torch.Tensor,
        text_unit_loss: torch.Tensor,
        visual_unit_loss: torch.Tensor,
        gate_probability: torch.Tensor,
        consistency: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        utility_loss, budget_loss, utility_target = saver_auxiliary_losses(
            gate_probability,
            text_unit_loss,
            visual_unit_loss,
            valid_mask,
            self.config.target_activation,
            self.config.utility_margin,
        )
        consistency_loss = _masked_mean(consistency, valid_mask)
        total = (
            task_loss
            + self.config.lambda_utility * utility_loss
            + self.config.lambda_budget * budget_loss
            + self.config.lambda_consistency * consistency_loss
        )
        losses = {
            "task": task_loss,
            "utility": utility_loss,
            "budget": budget_loss,
            "consistency": consistency_loss,
            "total": total,
        }
        return total, losses, utility_target


class SaverForMRE(_SaverTaskBase):
    """SAVER relation classifier with a marked entity-pair routing unit."""

    def __init__(
        self,
        config: SaverConfig,
        text_encoder: nn.Module,
        vision_encoder: Siglip2VisionEncoder,
        num_relations: int,
        text_hidden_size: Optional[int] = None,
        vision_hidden_size: Optional[int] = None,
    ) -> None:
        text_hidden = int(
            text_hidden_size
            if text_hidden_size is not None
            else getattr(text_encoder, "hidden_size")
        )
        vision_hidden = int(
            vision_hidden_size
            if vision_hidden_size is not None
            else getattr(vision_encoder, "hidden_size")
        )
        super().__init__(
            config,
            text_encoder,
            vision_encoder,
            text_hidden,
            vision_hidden,
        )
        self.pair_encoder = EntityPairRepresentation(
            text_hidden,
            config.max_span_width,
            config.width_embedding_size,
            config.task_hidden_size,
            config.dropout,
        )
        self.classifier = nn.Linear(config.task_hidden_size, num_relations)

    @classmethod
    def from_pretrained(
        cls,
        config: SaverConfig,
        num_relations: int,
        **kwargs: Any,
    ) -> "SaverForMRE":
        text_encoder = ModernBertTextEncoder.from_pretrained(
            config.text_model_name,
            freeze=config.freeze_text_encoder,
            attn_implementation=config.attn_implementation,
            **kwargs,
        )
        vision_encoder = Siglip2VisionEncoder.from_pretrained(
            config.vision_model_name,
            freeze=config.freeze_vision_encoder,
            attn_implementation=config.attn_implementation,
            **kwargs,
        )
        return cls(config, text_encoder, vision_encoder, num_relations)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        head_spans: torch.Tensor,
        tail_spans: torch.Tensor,
        global_inputs: TensorDict,
        detail_inputs: TensorDict,
        image_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gate_type: str = "learned",
        routing_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        gate_floor: float = 0.0,
        **text_kwargs: torch.Tensor,
    ) -> SaverOutput:
        mode = routing_mode or ("soft" if self.training else "hard")
        token_states = self.text_encoder(
            input_ids,
            attention_mask,
            **text_kwargs,
        )
        pair_state = self.pair_encoder(token_states, head_spans, tail_spans)
        text_logits = self.classifier(pair_state)
        unit_state = pair_state.unsqueeze(1)
        unit_mask = torch.ones(
            pair_state.size(0),
            1,
            dtype=torch.bool,
            device=pair_state.device,
        )
        routed = self._route_and_pool(
            unit_state,
            text_logits.unsqueeze(1),
            unit_mask,
            global_inputs,
            detail_inputs,
            image_mask,
            gate_type,
            mode,
            threshold,
            gate_floor,
        )
        fused = self.fusion(unit_state, routed["evidence"], routed["mix"])
        visual = self.fusion(
            unit_state,
            routed["evidence"],
            routed["has_evidence"].to(pair_state.dtype),
        )
        logits = self.classifier(fused.squeeze(1))
        visual_logits = self.classifier(visual.squeeze(1))

        total_loss = None
        losses: Dict[str, torch.Tensor] = {}
        utility_target = None
        if labels is not None:
            task_loss = F.cross_entropy(logits, labels)
            text_unit_loss = F.cross_entropy(
                text_logits,
                labels,
                reduction="none",
            ).unsqueeze(1)
            visual_unit_loss = F.cross_entropy(
                visual_logits,
                labels,
                reduction="none",
            ).unsqueeze(1)
            total_loss, losses, utility_target = self._combine_losses(
                task_loss,
                text_unit_loss,
                visual_unit_loss,
                routed["gate_probability"],
                routed["consistency"],
                unit_mask,
            )
        return SaverOutput(
            loss=total_loss,
            logits=logits,
            text_logits=text_logits,
            visual_logits=visual_logits,
            gate_probability=routed["gate_probability"].squeeze(1),
            gate_active=routed["gate_active"].squeeze(1),
            selected_indices=routed["selected_indices"].squeeze(1),
            attachment_relevance=routed["attachment_relevance"].squeeze(1),
            selected_regions=routed["selected_regions"].squeeze(1),
            utility_target=(
                None if utility_target is None else utility_target.squeeze(1)
            ),
            losses=losses,
        )


class SaverForMNER(_SaverTaskBase):
    """Span-enumeration MNER with per-span conditional visual evidence."""

    def __init__(
        self,
        config: SaverConfig,
        text_encoder: nn.Module,
        vision_encoder: Siglip2VisionEncoder,
        num_entity_labels: int,
        text_hidden_size: Optional[int] = None,
        vision_hidden_size: Optional[int] = None,
    ) -> None:
        text_hidden = int(
            text_hidden_size
            if text_hidden_size is not None
            else getattr(text_encoder, "hidden_size")
        )
        vision_hidden = int(
            vision_hidden_size
            if vision_hidden_size is not None
            else getattr(vision_encoder, "hidden_size")
        )
        super().__init__(
            config,
            text_encoder,
            vision_encoder,
            text_hidden,
            vision_hidden,
        )
        self.span_encoder = SpanRepresentation(
            text_hidden,
            config.max_span_width,
            config.width_embedding_size,
            config.task_hidden_size,
            config.dropout,
        )
        self.classifier = nn.Linear(config.task_hidden_size, num_entity_labels)

    @classmethod
    def from_pretrained(
        cls,
        config: SaverConfig,
        num_entity_labels: int,
        **kwargs: Any,
    ) -> "SaverForMNER":
        text_encoder = ModernBertTextEncoder.from_pretrained(
            config.text_model_name,
            freeze=config.freeze_text_encoder,
            attn_implementation=config.attn_implementation,
            **kwargs,
        )
        vision_encoder = Siglip2VisionEncoder.from_pretrained(
            config.vision_model_name,
            freeze=config.freeze_vision_encoder,
            attn_implementation=config.attn_implementation,
            **kwargs,
        )
        return cls(config, text_encoder, vision_encoder, num_entity_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        spans: torch.Tensor,
        span_mask: torch.Tensor,
        span_widths: torch.Tensor,
        global_inputs: TensorDict,
        detail_inputs: TensorDict,
        image_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gate_type: str = "learned",
        routing_mode: Optional[str] = None,
        threshold: Optional[float] = None,
        gate_floor: float = 0.0,
        **text_kwargs: torch.Tensor,
    ) -> SaverOutput:
        mode = routing_mode or ("soft" if self.training else "hard")
        token_states = self.text_encoder(
            input_ids,
            attention_mask,
            **text_kwargs,
        )
        span_state = self.span_encoder(token_states, spans, span_widths)
        text_logits = self.classifier(span_state)
        routed = self._route_and_pool(
            span_state,
            text_logits,
            span_mask,
            global_inputs,
            detail_inputs,
            image_mask,
            gate_type,
            mode,
            threshold,
            gate_floor,
        )
        fused = self.fusion(span_state, routed["evidence"], routed["mix"])
        visual = self.fusion(
            span_state,
            routed["evidence"],
            routed["has_evidence"].to(span_state.dtype),
        )
        logits = self.classifier(fused)
        visual_logits = self.classifier(visual)

        total_loss = None
        losses: Dict[str, torch.Tensor] = {}
        utility_target = None
        if labels is not None:
            valid = span_mask.bool() & labels.ne(-100)
            flat_labels = labels.reshape(-1)
            task_per_unit = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                flat_labels,
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            task_loss = _masked_mean(task_per_unit, valid)
            text_unit_loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                flat_labels,
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            visual_unit_loss = F.cross_entropy(
                visual_logits.reshape(-1, visual_logits.size(-1)),
                flat_labels,
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            total_loss, losses, utility_target = self._combine_losses(
                task_loss,
                text_unit_loss,
                visual_unit_loss,
                routed["gate_probability"],
                routed["consistency"],
                valid,
            )
        return SaverOutput(
            loss=total_loss,
            logits=logits,
            text_logits=text_logits,
            visual_logits=visual_logits,
            gate_probability=routed["gate_probability"],
            gate_active=routed["gate_active"],
            selected_indices=routed["selected_indices"],
            attachment_relevance=routed["attachment_relevance"],
            selected_regions=routed["selected_regions"],
            utility_target=utility_target,
            losses=losses,
        )

    @staticmethod
    def decode_non_overlapping(
        logits: torch.Tensor,
        spans: torch.Tensor,
        span_mask: torch.Tensor,
        none_label_id: int = 0,
    ) -> List[List[Dict[str, Any]]]:
        """Greedily decode non-overlapping typed spans by confidence."""

        probabilities = F.softmax(logits, dim=-1)
        scores, labels = probabilities.max(-1)
        decoded: List[List[Dict[str, Any]]] = []
        for batch_index in range(logits.size(0)):
            candidates: List[Tuple[float, int, int, int]] = []
            for span_index in range(logits.size(1)):
                if not bool(span_mask[batch_index, span_index]):
                    continue
                label = int(labels[batch_index, span_index])
                if label == none_label_id:
                    continue
                start, end = [
                    int(value) for value in spans[batch_index, span_index].tolist()
                ]
                candidates.append(
                    (
                        float(scores[batch_index, span_index]),
                        start,
                        end,
                        label,
                    )
                )
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            accepted: List[Dict[str, Any]] = []
            occupied: List[Tuple[int, int]] = []
            for score, start, end, label in candidates:
                overlaps = any(
                    start < other_end and other_start < end
                    for other_start, other_end in occupied
                )
                if overlaps:
                    continue
                occupied.append((start, end))
                accepted.append(
                    {
                        "start": start,
                        "end": end,
                        "label": label,
                        "score": score,
                    }
                )
            accepted.sort(key=lambda item: (item["start"], item["end"]))
            decoded.append(accepted)
        return decoded


class RiskControlledCalibrator:
    """Finite-grid calibration with simultaneous Clopper--Pearson bounds.

    A fixed threshold grid is chosen before inspecting calibration outcomes.
    Bonferroni correction over that grid avoids treating an adaptively selected
    pointwise confidence bound as a simultaneous guarantee.
    """

    def __init__(
        self,
        target_risk: float = 0.10,
        confidence: float = 0.95,
        grid_size: int = 101,
    ) -> None:
        if not 0.0 < target_risk < 1.0:
            raise ValueError("target_risk must be in (0, 1)")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        self.target_risk = target_risk
        self.confidence = confidence
        self.grid_size = grid_size

    @staticmethod
    def _binomial_cdf(k: int, n: int, probability: float) -> float:
        if k < 0:
            return 0.0
        if k >= n:
            return 1.0
        if probability <= 0.0:
            return 1.0
        if probability >= 1.0:
            return 0.0
        logs = [
            math.lgamma(n + 1)
            - math.lgamma(index + 1)
            - math.lgamma(n - index + 1)
            + index * math.log(probability)
            + (n - index) * math.log1p(-probability)
            for index in range(k + 1)
        ]
        maximum = max(logs)
        return math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)

    @classmethod
    def _clopper_pearson_upper(cls, k: int, n: int, alpha: float) -> float:
        if n <= 0 or k >= n:
            return 1.0
        if k <= 0:
            return 1.0 - alpha ** (1.0 / n)
        try:
            from scipy.stats import beta

            return float(beta.ppf(1.0 - alpha, k + 1, n - k))
        except ImportError:
            lower, upper = 0.0, 1.0
            for _ in range(64):
                midpoint = (lower + upper) / 2.0
                if cls._binomial_cdf(k, n, midpoint) > alpha:
                    lower = midpoint
                else:
                    upper = midpoint
            return upper

    def calibrate(
        self,
        scores: torch.Tensor,
        harmful: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        scores = scores.detach().reshape(-1).float().cpu()
        harmful = harmful.detach().reshape(-1).bool().cpu()
        if valid_mask is not None:
            valid = valid_mask.detach().reshape(-1).bool().cpu()
            scores = scores[valid]
            harmful = harmful[valid]
        if scores.numel() == 0:
            return self._fallback()

        delta = 1.0 - self.confidence
        per_threshold_alpha = delta / self.grid_size
        feasible: List[Dict[str, float]] = []
        total = int(scores.numel())
        for threshold_tensor in torch.linspace(0.0, 1.0, self.grid_size):
            threshold = float(threshold_tensor)
            active = scores >= threshold
            activated = int(active.sum())
            if activated == 0:
                continue
            failures = int(harmful[active].sum())
            upper = self._clopper_pearson_upper(
                failures,
                activated,
                per_threshold_alpha,
            )
            if upper <= self.target_risk:
                feasible.append(
                    {
                        "threshold": threshold,
                        "coverage": activated / total,
                        "empirical_risk": failures / activated,
                        "risk_upper": upper,
                        "activated": float(activated),
                        "harmful": float(failures),
                        "grid_size": float(self.grid_size),
                        "per_threshold_alpha": per_threshold_alpha,
                    }
                )
        if not feasible:
            return self._fallback()
        return max(
            feasible,
            key=lambda item: (item["coverage"], item["threshold"]),
        )

    def _fallback(self) -> Dict[str, float]:
        return {
            "threshold": 1.000001,
            "coverage": 0.0,
            "empirical_risk": 0.0,
            "risk_upper": 1.0,
            "activated": 0.0,
            "harmful": 0.0,
            "grid_size": float(self.grid_size),
            "per_threshold_alpha": (1.0 - self.confidence) / self.grid_size,
        }


def build_saver_model(
    task: str,
    config: SaverConfig,
    num_labels: int,
    **pretrained_kwargs: Any,
) -> nn.Module:
    """Construct a pretrained MRE or MNER SAVER model."""

    normalized = task.lower()
    if normalized == "mre":
        return SaverForMRE.from_pretrained(
            config,
            num_relations=num_labels,
            **pretrained_kwargs,
        )
    if normalized == "mner":
        return SaverForMNER.from_pretrained(
            config,
            num_entity_labels=num_labels,
            **pretrained_kwargs,
        )
    raise ValueError("task must be 'mre' or 'mner'")
