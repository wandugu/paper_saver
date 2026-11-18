"""SAVER Section 3 task model over detached, frozen-backbone features.

One forward handles the units of one post and its shared attachment bank.
``head`` and ``tail`` contain [first; last; interior-max] token features;
width embeddings and all task/query/vision projections are learned here.
Call ``eval()`` for deterministic inference. ``training`` overrides routing
semantics only, permitting soft-routed development evaluation in eval mode.
"""

from collections.abc import Callable, Mapping, Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def utility_targets(text_losses: Tensor, vision_losses: Tensor, margin: float = 0.05) -> Tensor:
    """Return detached Eq. (utility_target) labels, with a strict inequality."""
    return (vision_losses.detach() + margin < text_losses.detach()).to(text_losses.dtype)


def gate_floor(step: int, forced_steps: int = 500, anneal_steps: int = 500) -> float:
    """Zero-based optimizer-step schedule from the supplementary protocol."""
    if step < 0 or forced_steps < 0 or anneal_steps < 0:
        raise ValueError("Optimizer step and schedule lengths must be nonnegative")
    if step < forced_steps:
        return 1.0
    if not anneal_steps:
        return 0.0
    return max(0.0, 1.0 - (step - forced_steps) / anneal_steps)


class SaverCore(nn.Module):
    """Trainable task head for frozen ModernBERT/SigLIP(-2) features.

    ``units`` is a mapping with ``head: [U, 3*text_dim]`` and
    ``head_width: [U]``. MRE additionally requires ``tail`` and ``tail_width``.
    Optional ``mask: [U]`` excludes padded units from routing and losses.
    ``global_features: [N, vision_dim]`` preserves manifest order;
    ``global_mask: [N]`` excludes invalid/padded attachments.

    ``details`` maps attachment indices to ``[patches, vision_dim]`` tensors.
    Alternatively, ``detail_provider(indices)`` returns that mapping, a list
    of tensors in index order, or a batched tensor. It is called once for the
    union of missing required indices. Supplied features are always detached.
    ``always_vision`` requires every valid attachment's high-resolution
    features, but task evidence still pools only the Top-1 attachment.

    ``force_details`` materializes forced-vision logits for auditing without
    changing routed activation. Training always materializes this branch for
    utility labels. ``high_resolution_calls`` counts distinct required detail
    encodings, including cache hits; ``detail_cache_misses`` counts provider
    requests. Cold/hot encoder timing is the runner's responsibility.
    """

    def __init__(
        self,
        num_labels: int,
        *,
        task: str = "mre",
        text_dim: int = 768,
        vision_dim: int = 768,
        hidden_dim: int = 384,
        query_dim: int = 256,
        width_dim: int = 32,
        max_span_width: int = 128,
        num_heads: int = 4,
        top_k: int = 1,
        top_r: int = 3,
        pooling: str = "attention",
        dropout: float = 0.1,
        forced_steps: int = 500,
        anneal_steps: int = 500,
        rho: float = 0.35,
        utility_margin: float = 0.05,
        text_ce_weight: float = 0.0,
        route_dropout: float = 0.0,
        anchor_residual: bool = False,
    ):
        super().__init__()
        if task.lower() not in {"mre", "mner"}:
            raise ValueError("task must be mre or mner")
        if min(num_labels, text_dim, vision_dim, hidden_dim, query_dim, width_dim,
               max_span_width, num_heads, top_k, top_r) <= 0:
            raise ValueError("Dimensions, widths and retrieval budgets must be positive")
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be attention or mean")
        if not 0 <= rho <= 1 or utility_margin < 0:
            raise ValueError("Invalid activation target or utility margin")
        gate_floor(0, forced_steps, anneal_steps)
        self.task = task.lower()
        self.text_dim, self.vision_dim = text_dim, vision_dim
        self.hidden_dim, self.query_dim = hidden_dim, query_dim
        self.max_span_width = max_span_width
        self.top_k, self.top_r, self.pooling = top_k, top_r, pooling
        self.forced_steps, self.anneal_steps = forced_steps, anneal_steps
        self.rho, self.utility_margin = rho, utility_margin
        if text_ce_weight < 0 or not 0 <= route_dropout <= 1:
            raise ValueError("Invalid phase5 text loss or route dropout")
        self.text_ce_weight, self.route_dropout = text_ce_weight, route_dropout
        self.width_embedding = nn.Embedding(max_span_width + 1, width_dim, padding_idx=0)
        self.span_projection = nn.Linear(3 * text_dim + width_dim, hidden_dim)
        self.pair_projection = nn.Linear(4 * hidden_dim, hidden_dim) if self.task == "mre" else None
        self.query_projection = nn.Linear(hidden_dim, query_dim)
        self.global_projection = nn.Linear(vision_dim, query_dim)
        self.patch_projection = nn.Linear(vision_dim, query_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * query_dim + 4, query_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(query_dim, 1),
        )
        self.patch_attention = nn.MultiheadAttention(
            query_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.evidence_projection = nn.Linear(query_dim, hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.consistency_query = nn.Linear(query_dim, query_dim)
        self.consistency_evidence = nn.Linear(query_dim, query_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.visual_residual_head = nn.Linear(hidden_dim, num_labels) if anchor_residual else None

    def _span(self, units: Mapping[str, Tensor], name: str, mask: Tensor) -> Tensor:
        raw = units[name].detach().to(self.span_projection.weight)
        if raw.ndim != 2 or raw.shape != (mask.numel(), 3 * self.text_dim):
            raise ValueError(f"{name} must have shape [U, {3 * self.text_dim}]")
        widths = torch.as_tensor(units[f"{name}_width"], device=raw.device)
        if widths.shape != mask.shape or widths.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name}_width must be an integer [U] tensor")
        if ((widths[mask] < 1) | (widths[mask] > self.max_span_width)).any():
            raise ValueError(f"{name}_width is outside [1, {self.max_span_width}]")
        raw = raw.masked_fill(~mask[:, None], 0)
        widths = widths.long().masked_fill(~mask, 0)
        return self.span_projection(torch.cat((raw, self.width_embedding(widths)), dim=-1))

    @staticmethod
    def _method(method: str) -> str:
        method = method.lower().replace("-", "_").replace("+", "_")
        aliases = {"text": "text_only", "always": "always_vision", "maxsim_top_1": "maxsim",
                   "maxsim_top1": "maxsim", "random_gate": "random", "mean_pool": "mean"}
        method = aliases.get(method, method)
        if method not in {"saver", "text_only", "always_vision", "maxsim", "random", "mean", "topk2"}:
            raise ValueError(f"Unknown routing method: {method}")
        return method

    def _details(self, indices: list[int], details, detail_provider):
        bank = {} if details is None else {index: details[index] for index in indices if index in details}
        missing = [index for index in indices if index not in bank]
        if missing:
            if detail_provider is None:
                raise ValueError(f"Missing high-resolution patch features for indices {missing}")
            supplied = detail_provider(missing)
            if isinstance(supplied, Mapping):
                bank.update({index: supplied[index] for index in missing})
            elif isinstance(supplied, Tensor):
                if supplied.ndim == 2 and len(missing) == 1:
                    bank[missing[0]] = supplied
                elif supplied.ndim == 3 and supplied.shape[0] == len(missing):
                    bank.update(zip(missing, supplied.unbind(0)))
                else:
                    raise ValueError("detail_provider returned a tensor with incompatible shape")
            elif isinstance(supplied, Sequence) and len(supplied) == len(missing):
                bank.update(zip(missing, supplied))
            else:
                raise ValueError("detail_provider must return a mapping, ordered sequence or tensor")
        projected = {}
        for index, features in bank.items():
            features = features.detach().to(self.patch_projection.weight)
            if features.ndim != 2 or features.shape[1] != self.vision_dim or not features.shape[0]:
                raise ValueError("Detail features must be nonempty [patches, vision_dim] tensors")
            projected[index] = F.normalize(self.patch_projection(features), dim=-1)
        return projected, len(missing)

    def _pool(self, queries: Tensor, globals_: Tensor, selected: Tensor, bank, pooling: str):
        unit_count, selected_count = selected.shape
        patch_count = max((features.shape[0] for features in bank.values()), default=0)
        record_count = patch_count if pooling == "mean" else min(self.top_r, patch_count)
        patchids = selected.new_full((unit_count, selected_count, record_count), -1)
        chunks = [[] for _ in range(unit_count)]
        weights = [[] for _ in range(unit_count)]
        for image_index, patches in bank.items():
            locations = (selected == image_index).nonzero(as_tuple=False)
            unit_indices = locations[:, 0]
            if pooling == "attention":
                scores = queries.index_select(0, unit_indices) @ patches.T
                ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :self.top_r]
                gathered = patches[ranked]
                tokens = torch.cat((globals_[image_index].view(1, 1, -1).expand(
                    len(locations), 1, -1), gathered), dim=1)
                for local, (unit_index, slot) in enumerate(locations.tolist()):
                    patchids[unit_index, slot, :ranked.shape[1]] = ranked[local]
                    chunks[unit_index].append(tokens[local])
            else:
                pooled = (globals_[image_index] + patches.sum(0)) / (patches.shape[0] + 1)
                for unit_index, slot in locations.tolist():
                    patchids[unit_index, slot, :patches.shape[0]] = torch.arange(
                        patches.shape[0], device=selected.device)
                    chunks[unit_index].append(pooled)
                    weights[unit_index].append(patches.shape[0] + 1)
        available = (selected >= 0).any(dim=-1)
        available_indices = available.nonzero(as_tuple=False).flatten()
        evidence = queries.new_zeros((unit_count, self.query_dim))
        if available_indices.numel():
            if pooling == "attention":
                sequences = [torch.cat(chunks[index], dim=0) for index in available_indices.tolist()]
                keys = pad_sequence(sequences, batch_first=True)
                lengths = selected.new_tensor([sequence.shape[0] for sequence in sequences])
                padding = torch.arange(keys.shape[1], device=keys.device)[None, :] >= lengths[:, None]
                pooled, _ = self.patch_attention(
                    queries[available_indices, None], keys, keys,
                    key_padding_mask=padding, need_weights=False,
                )
                pooled = pooled.squeeze(1)
            else:
                pooled = torch.stack([
                    (torch.stack(chunks[index]) * queries.new_tensor(weights[index])[:, None]).sum(0)
                    / sum(weights[index]) for index in available_indices.tolist()
                ])
            evidence = evidence.index_copy(0, available_indices, pooled)
        return evidence, patchids, available

    def forward(
        self,
        units: Mapping[str, Tensor],
        global_features: Tensor,
        details: Mapping[int, Tensor] | None = None,
        *,
        detail_provider: Callable | None = None,
        method: str = "saver",
        step: int = 0,
        threshold: float = 0.5,
        training: bool | None = None,
        force_details: bool = False,
        random_draws: Tensor | None = None,
        random_rate: float | None = None,
        global_mask: Tensor | None = None,
        top_k: int | None = None,
        pooling: str | None = None,
        labels: Tensor | None = None,
        class_weights: Tensor | None = None,
    ) -> dict:
        """Return logits, auditable routing tensors and optional supervised losses.

        ``selected`` records acquired image indices, padded with -1;
        ``retrieved`` also records gate-off candidate rankings. ``patchids``
        is [U, K, R] (all valid patches for mean pooling), padded with -1.
        Random routing requires runner-supplied U(0,1) draws and a frozen
        activation-matching ``random_rate``. It never samples internally.
        """
        method = self._method(method)
        training = self.training if training is None else training
        threshold = float(threshold)
        if math.isnan(threshold) or threshold < 0:
            raise ValueError("threshold must be nonnegative (positive infinity disables routing)")
        floor = gate_floor(step, self.forced_steps, self.anneal_steps)
        device = self.span_projection.weight.device
        unit_count = units["head"].shape[0]
        mask = torch.as_tensor(units.get("mask", torch.ones(unit_count, dtype=torch.bool)),
                               device=device, dtype=torch.bool)
        if mask.shape != (unit_count,):
            raise ValueError("units['mask'] must have shape [U]")
        head = self._span(units, "head", mask)
        if self.task == "mre":
            tail = self._span(units, "tail", mask)
            text_state = self.pair_projection(torch.cat((head, tail, head * tail, (head - tail).abs()), -1))
        else:
            text_state = head
        text_logits = self.classifier(text_state)
        queries = F.normalize(self.query_projection(text_state), dim=-1)
        globals_raw = global_features.detach().to(self.global_projection.weight)
        if globals_raw.ndim != 2 or globals_raw.shape[1] != self.vision_dim:
            raise ValueError("global_features must have shape [N, vision_dim]")
        image_count = globals_raw.shape[0]
        image_mask = torch.ones(image_count, device=device, dtype=torch.bool) if global_mask is None else (
            torch.as_tensor(global_mask, device=device, dtype=torch.bool))
        if image_mask.shape != (image_count,):
            raise ValueError("global_mask must have shape [N]")
        eligible = mask & image_mask.any()
        use_vision = method != "text_only" and bool(eligible.any())
        count = self.top_k if top_k is None else top_k
        count = 2 if method == "topk2" else count
        count = 1 if method == "always_vision" else count
        if not isinstance(count, int) or count < 1:
            raise ValueError("top_k must be a positive integer")
        pooling = ("mean" if method == "mean" else self.pooling) if pooling is None else pooling
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be attention or mean")
        gate = queries.new_zeros(unit_count)
        eta = gate
        statistics = queries.new_zeros((unit_count, 4))
        similarities = queries.new_full((unit_count, image_count), -torch.inf)
        globals_projected = queries.new_zeros((image_count, self.query_dim))
        retrieved = torch.full((unit_count, count), -1, device=device, dtype=torch.long)
        if use_vision:
            globals_projected = F.normalize(self.global_projection(globals_raw.masked_fill(
                ~image_mask[:, None], 0)), dim=-1)
            similarities = (queries @ globals_projected.T).masked_fill(~image_mask[None, :], -torch.inf)
            ranking = torch.argsort(similarities, dim=-1, descending=True, stable=True)
            slots = min(count, image_count)
            candidates = ranking[:, :slots]
            retrieved[:, :slots] = candidates.masked_fill(~image_mask[candidates] | ~mask[:, None], -1)
            valid_scores = similarities[:, image_mask]
            maximum = valid_scores.max(-1).values
            statistics = torch.stack((maximum, valid_scores.mean(-1), valid_scores.std(-1, unbiased=False),
                                      valid_scores.sort(dim=-1, descending=True).values[:, :2].mean(-1)), -1)
            best = globals_projected[ranking[:, 0]]
            if method in {"saver", "mean", "topk2"}:
                interaction = torch.cat((queries, best, queries * best, (queries - best).abs(), statistics), -1)
                gate = self.gate_mlp(interaction).squeeze(-1).sigmoid()
                eta = floor + (1 - floor) * gate if training else (gate >= threshold).to(gate.dtype)
            elif method == "maxsim":
                gate = ((1 + maximum) / 2).clamp(0, 1)
                eta = (gate >= threshold).to(gate.dtype)
            elif method == "always_vision":
                gate = torch.ones_like(gate)
                eta = gate
            else:
                if random_rate is None or not 0 <= random_rate <= 1 or random_draws is None:
                    raise ValueError("random requires runner-supplied random_draws and random_rate in [0,1]")
                draws = torch.as_tensor(random_draws, device=device, dtype=queries.dtype).detach()
                if draws.shape != (unit_count,) or not torch.isfinite(draws).all() or ((draws < 0) | (draws >= 1)).any():
                    raise ValueError("random_draws must be finite [U] values in [0,1)")
                gate = torch.full_like(gate, random_rate)
                eta = (draws < random_rate).to(gate.dtype)
            gate = gate.masked_fill(~eligible, 0)
            eta = eta.masked_fill(~eligible, 0)
            if training and self.route_dropout and method in {"saver", "mean", "topk2"}:
                eta = eta.masked_fill(torch.rand_like(eta) < self.route_dropout, 0)
        active = (eta > 0) & eligible
        acquire = eligible & (active | bool(training) | force_details) if use_vision else torch.zeros_like(mask)
        selected = retrieved.masked_fill(~acquire[:, None], -1)
        required = sorted(set(selected[selected >= 0].tolist()))
        if method == "always_vision" and use_vision:
            required = image_mask.nonzero(as_tuple=False).flatten().tolist()
        bank, misses = self._details(required, details, detail_provider)
        evidence, patchids, available = self._pool(queries, globals_projected, selected, bank, pooling)
        visual_state = text_state
        if available.any():
            indices = available.nonzero(as_tuple=False).flatten()
            base = text_state[indices]
            projected = self.evidence_projection(evidence[indices])
            visual = self.fusion_mlp(torch.cat((base, projected, base * projected, (base - projected).abs()), -1))
            visual_state = text_state.index_copy(0, indices, visual)
        vision_logits = self.classifier(visual_state)
        routed_state = text_state + eta[:, None] * (visual_state - text_state)
        logits = torch.where((eta == 0)[:, None], text_logits, self.classifier(routed_state))
        if self.visual_residual_head is not None:
            residual = self.visual_residual_head(visual_state)
            residual = residual.masked_fill(~available[:, None], 0)
            vision_logits = text_logits + residual
            logits = torch.where((eta == 0)[:, None], text_logits, text_logits + eta[:, None] * residual)
        output = {
            "text_logits": text_logits, "vision_logits": vision_logits, "logits": logits,
            "gate": gate, "eta": eta, "active": active, "selected": selected,
            "retrieved": retrieved, "patchids": patchids, "evidence": evidence,
            "query": queries, "similarities": similarities, "similarity_stats": statistics,
            "text_state": text_state, "vision_state": visual_state, "routed_state": routed_state,
            "vision_available": available, "unit_mask": mask, "image_available": eligible,
            "global_screen_calls": int(image_mask.sum()) if method != "text_only" else 0,
            "high_resolution_calls": len(required), "detail_cache_misses": misses,
            "method": method,
        }
        if labels is not None:
            output.update(self.compute_losses(output, labels, class_weights=class_weights))
        return output

    def compute_losses(self, output: dict, labels: Tensor, *, class_weights: Tensor | None = None) -> dict:
        """Task CE plus .25 utility, .05 budget and .05 consistency.

        Padded/unlabelled units are excluded. Gate auxiliary terms are used
        only by SAVER and its pooling/K ablations, never by fixed controls.
        The consistency term is shared by visual controls. No-image units
        contribute text CE only, since their visual branch does not exist.
        """
        labels = torch.as_tensor(labels, device=output["logits"].device, dtype=torch.long)
        if labels.shape != output["unit_mask"].shape:
            raise ValueError("labels must have shape [U]")
        valid = output["unit_mask"] & (labels != -100)
        safe_labels = labels.masked_fill(~valid, -100)
        weights = None if class_weights is None else class_weights.to(output["logits"])
        task_ce = F.cross_entropy(output["logits"], safe_labels, weight=weights, reduction="none", ignore_index=-100)
        text_ce = F.cross_entropy(output["text_logits"], safe_labels, weight=weights, reduction="none", ignore_index=-100)
        vision_ce = F.cross_entropy(output["vision_logits"], safe_labels, weight=weights, reduction="none", ignore_index=-100)
        zero = output["logits"].sum() * 0
        task_loss = task_ce[valid].mean() if valid.any() else zero
        targets = utility_targets(text_ce, vision_ce, self.utility_margin)
        visual = valid & output["vision_available"]
        eligible = valid & output["image_available"]
        utility_loss, budget_loss, consistency_loss = zero, zero, zero
        if output["method"] in {"saver", "mean", "topk2"}:
            if visual.any():
                utility_loss = F.binary_cross_entropy(output["gate"][visual], targets[visual])
            if eligible.any():
                budget_loss = (output["gate"][eligible].mean() - self.rho).square()
        if visual.any():
            query = self.consistency_query(output["query"][visual])
            evidence = self.consistency_evidence(output["evidence"][visual])
            consistency_loss = (1 - F.cosine_similarity(query, evidence, dim=-1)).mean()
        text_loss = text_ce[valid].mean() if valid.any() else zero
        loss = task_loss + self.text_ce_weight * text_loss + 0.25 * utility_loss + 0.05 * budget_loss + 0.05 * consistency_loss
        return {"loss": loss, "task_loss": task_loss, "utility_loss": utility_loss,
                "budget_loss": budget_loss, "consistency_loss": consistency_loss,
                "utility_targets": targets, "text_losses": text_ce, "vision_losses": vision_ce,
                "text_loss": text_loss}
