import torch

from models.saver_components import GlobalGroundabilityGate, MiniSetTransformer, SISSelector, SaverFusion


def main():
    torch.manual_seed(7)
    bsz, n_img, text_dim, vis_dim = 1, 3, 8, 6
    span = torch.randn(bsz, text_dim)
    image_global = torch.randn(bsz, n_img, vis_dim)

    cgg = GlobalGroundabilityGate(text_dim=text_dim, vision_dim=vis_dim, hidden_dim=16)
    score, detail = cgg(span, image_global)
    assert score.shape == (bsz,)
    assert "top2_mean" in detail

    sis = SISSelector(lambda_rel=1.0, lambda_cov=1.0)
    result = sis.select(query=torch.randn(vis_dim), image_global=image_global[0], budget_k=2)
    assert len(result.selected_indices) <= 2

    set_pool = MiniSetTransformer(dim=vis_dim, heads=2)
    selected = image_global[:, result.selected_indices or [0], :]
    pooled = set_pool(selected)
    assert pooled.shape == (bsz, vis_dim)

    fusion = SaverFusion(text_dim=text_dim, vision_dim=vis_dim)
    fused = fusion(span, pooled, score)
    assert fused.shape == (bsz, text_dim)
    print("SAVER组件自测通过，selected=", result.selected_indices, "objective=", result.objective)


if __name__ == "__main__":
    main()
