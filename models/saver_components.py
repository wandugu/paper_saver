import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class SISSelectionResult:
    selected_indices: List[int]
    objective: float
    relevance_scores: List[float]


class GlobalGroundabilityGate(nn.Module):
    """CGG: 使用全局图像向量计算可视可落地性分数。"""

    def __init__(self, text_dim: int, vision_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, vision_dim)
        # 统计量: max / mean / std / top2_mean
        self.scorer = nn.Sequential(
            nn.Linear(text_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _cosine_sim(query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        query = nn.functional.normalize(query, dim=-1)
        values = nn.functional.normalize(values, dim=-1)
        return torch.einsum("bd,bnd->bn", query, values)

    def forward(self, span_repr: torch.Tensor, image_global: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        query = self.query_proj(span_repr)
        sims = self._cosine_sim(query, image_global)
        max_sim, _ = sims.max(dim=-1)
        mean_sim = sims.mean(dim=-1)
        std_sim = sims.std(dim=-1, unbiased=False)
        topk = min(2, sims.size(1))
        topk_mean = sims.topk(k=topk, dim=-1).values.mean(dim=-1)
        stats = torch.stack([max_sim, mean_sim, std_sim, topk_mean], dim=-1)

        score = torch.sigmoid(self.scorer(torch.cat([span_repr, stats], dim=-1))).squeeze(-1)
        detail = {
            "max": max_sim,
            "mean": mean_sim,
            "std": std_sim,
            "top2_mean": topk_mean,
            "raw_sims": sims,
        }
        logger.debug("CGG统计量: max=%s mean=%s std=%s top2=%s", max_sim.tolist(), mean_sim.tolist(), std_sim.tolist(), topk_mean.tolist())
        return score, detail


class SISSelector:
    """SIS: relevance + facility-location 覆盖项。"""

    def __init__(self, lambda_rel: float = 1.0, lambda_cov: float = 1.0):
        self.lambda_rel = lambda_rel
        self.lambda_cov = lambda_cov

    @staticmethod
    def _rescale(sim: torch.Tensor) -> torch.Tensor:
        return (1.0 + sim) / 2.0

    def select(self, query: torch.Tensor, image_global: torch.Tensor, budget_k: int) -> SISSelectionResult:
        assert image_global.dim() == 2, "期望 [N, d]"
        q = nn.functional.normalize(query, dim=-1)
        v = nn.functional.normalize(image_global, dim=-1)
        rel = torch.mv(v, q)
        dmat = torch.mm(v, v.t())

        rel_tilde = self._rescale(rel)
        d_tilde = self._rescale(dmat)

        n = image_global.size(0)
        budget_k = min(max(0, budget_k), n)
        selected: List[int] = []
        current_max = torch.zeros(n, device=image_global.device)

        for step in range(budget_k):
            best_i = None
            best_gain = None
            for i in range(n):
                if i in selected:
                    continue
                gain = self.lambda_rel * rel_tilde[i]
                gain += self.lambda_cov * torch.sum(rel_tilde * torch.clamp(d_tilde[i] - current_max, min=0.0))
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            current_max = torch.maximum(current_max, d_tilde[best_i])
            logger.debug("SIS step=%d 选择图像=%d 增益=%.6f", step + 1, best_i, float(best_gain.item()))

        objective = 0.0
        if selected:
            selected_tensor = torch.tensor(selected, device=image_global.device)
            rel_part = self.lambda_rel * rel_tilde[selected_tensor].sum()
            cov_max = d_tilde[selected_tensor].max(dim=0).values
            cov_part = self.lambda_cov * torch.sum(rel_tilde * cov_max)
            objective = float((rel_part + cov_part).item())

        return SISSelectionResult(
            selected_indices=selected,
            objective=objective,
            relevance_scores=rel.tolist(),
        )


class MiniSetTransformer(nn.Module):
    """轻量 Set Transformer: 一层 SAB + PMA(seed=1)。"""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.sab = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.seed = nn.Parameter(torch.randn(1, 1, dim))
        self.pma = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        # evidence: [B, M, d]
        sab_out, _ = self.sab(evidence, evidence, evidence)
        sab_out = self.norm(sab_out + evidence)
        seed = self.seed.expand(evidence.size(0), -1, -1)
        pooled, _ = self.pma(seed, sab_out, sab_out)
        return pooled.squeeze(1)


class SaverFusion(nn.Module):
    def __init__(self, text_dim: int, vision_dim: int):
        super().__init__()
        self.fuse = nn.Linear(text_dim + vision_dim, text_dim)

    def forward(self, text_repr: torch.Tensor, vis_repr: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(torch.cat([text_repr, vis_repr], dim=-1))
        eta = eta.unsqueeze(-1)
        return (1 - eta) * text_repr + eta * fused
