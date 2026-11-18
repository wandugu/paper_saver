import math

import pytest
import torch
from torch import nn

from repro.p24.model import SaverCore, gate_floor, utility_targets


@pytest.fixture(autouse=True)
def deterministic_cpu():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        yield
    torch.set_num_threads(original_threads)


def make_model(task="mre", **kwargs):
    config = dict(text_dim=6, vision_dim=8, hidden_dim=12, query_dim=8,
                  width_dim=3, num_heads=4, dropout=0.0, forced_steps=4, anneal_steps=4)
    config.update(kwargs)
    return SaverCore(4, task=task, **config)


def make_inputs(unit_count=3, image_count=3):
    units = {"head": torch.randn(unit_count, 18), "head_width": torch.ones(unit_count, dtype=torch.long),
             "tail": torch.randn(unit_count, 18), "tail_width": torch.full((unit_count,), 2, dtype=torch.long)}
    globals_ = torch.randn(image_count, 8)
    details = {index: torch.randn(7, 8) for index in range(image_count)}
    return units, globals_, details


def gradient_sum(module):
    return sum(float(parameter.grad.abs().sum()) for parameter in module.parameters() if parameter.grad is not None)


def test_default_dimensions_and_classifier_sharing():
    model = SaverCore(7)
    assert model.span_projection.in_features == 3 * 768 + model.width_embedding.embedding_dim
    assert model.span_projection.out_features == 384
    assert model.query_projection.out_features == 256
    assert model.patch_attention.num_heads == 4
    assert model.classifier.out_features == 7
    assert model.forced_steps == model.anneal_steps == 500
    assert model.rho == 0.35


@pytest.mark.parametrize("step,expected", [(0, 1), (3, 1), (4, 1), (5, 0.75), (7, 0.25), (8, 0), (90, 0)])
def test_gate_floor_boundaries(step, expected):
    assert gate_floor(step, 4, 4) == expected


def test_gate_floor_no_schedule_and_negative_steps():
    assert gate_floor(0, 0, 0) == 0
    assert gate_floor(500) == 1
    assert gate_floor(750) == 0.5
    assert gate_floor(1000) == 0
    with pytest.raises(ValueError):
        gate_floor(-1)


def test_utility_targets_are_detached_with_strict_margin():
    text = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    visual = torch.tensor([0.9, 0.95, 1.1], requires_grad=True)
    targets = utility_targets(text, visual)
    assert torch.equal(targets, torch.tensor([1.0, 0.0, 0.0]))
    assert not targets.requires_grad
    assert targets.grad_fn is None


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("task", ["mre", "mner"])
def test_no_image_is_exact_text_and_skips_provider(training, task):
    model = make_model(task)
    units, globals_, _ = make_inputs(image_count=0)

    def forbidden(indices):
        pytest.fail(f"No-image branch requested detail features: {indices}")

    output = model(units, globals_, detail_provider=forbidden, training=training,
                   labels=torch.tensor([0, 1, 2]), force_details=True)
    assert torch.equal(output["logits"], output["text_logits"])
    assert torch.equal(output["vision_logits"], output["text_logits"])
    assert torch.equal(output["routed_state"], output["text_state"])
    assert output["evidence"].count_nonzero() == output["gate"].count_nonzero() == 0
    assert not output["active"].any()
    assert (output["selected"] == -1).all()
    assert output["high_resolution_calls"] == output["global_screen_calls"] == 0
    assert output["utility_loss"] == output["budget_loss"] == output["consistency_loss"] == 0
    assert torch.equal(output["loss"], output["task_loss"])
    assert torch.isfinite(output["loss"])


def test_gate_off_is_bitwise_text_fallback_and_no_detail_execution():
    model = make_model().eval()
    units, globals_, _ = make_inputs()
    output = model(units, globals_, threshold=math.inf)
    assert torch.equal(output["text_logits"], output["logits"])
    assert torch.equal(output["text_state"], output["routed_state"])
    assert output["high_resolution_calls"] == 0
    assert output["global_screen_calls"] == 3
    assert not output["active"].any()
    assert output["gate"].gt(0).all()


def test_force_details_audit_does_not_override_routed_gate():
    model = make_model().eval()
    units, globals_, details = make_inputs()
    output = model(units, globals_, details, threshold=math.inf, force_details=True)
    assert not output["active"].any()
    assert output["vision_available"].all()
    assert output["high_resolution_calls"] > 0
    assert torch.equal(output["logits"], output["text_logits"])
    assert not torch.equal(output["vision_logits"], output["text_logits"])


def test_training_eta_uses_exact_affine_floor_and_hard_inference():
    model = make_model()
    units, globals_, details = make_inputs()
    initial = model(units, globals_, details, step=0)
    annealed = model(units, globals_, details, step=6)
    final = model(units, globals_, details, step=8)
    assert torch.equal(initial["eta"], torch.ones(3))
    torch.testing.assert_close(annealed["eta"], 0.5 + 0.5 * annealed["gate"])
    assert torch.equal(final["eta"], final["gate"])
    assert final["vision_available"].all()
    deployed = model(units, globals_, details, training=False, threshold=0.5)
    assert torch.equal(deployed["eta"], (deployed["gate"] >= 0.5).float())


def test_training_gradients_reach_gate_fusion_and_projections_not_backbones():
    model = make_model()
    units, globals_, details = make_inputs()
    units["head"].requires_grad_()
    units["tail"].requires_grad_()
    globals_.requires_grad_()
    for patches in details.values():
        patches.requires_grad_()
    output = model(units, globals_, details, step=8, labels=torch.tensor([0, 1, 2]))
    expected = output["task_loss"] + 0.25 * output["utility_loss"] + 0.05 * output["budget_loss"] + 0.05 * output["consistency_loss"]
    torch.testing.assert_close(output["loss"], expected)
    assert not output["utility_targets"].requires_grad
    output["loss"].backward()
    for module in (model.gate_mlp, model.fusion_mlp, model.span_projection,
                   model.pair_projection, model.query_projection, model.global_projection,
                   model.patch_projection, model.width_embedding, model.patch_attention):
        assert gradient_sum(module) > 0
    assert units["head"].grad is units["tail"].grad is globals_.grad is None
    assert all(patches.grad is None for patches in details.values())


def test_task_loss_alone_trains_gate_after_floor_anneal():
    model = make_model()
    units, globals_, details = make_inputs()
    output = model(units, globals_, details, step=8, labels=torch.tensor([0, 1, 2]))
    output["task_loss"].backward()
    assert gradient_sum(model.gate_mlp) > 0
    assert gradient_sum(model.fusion_mlp) > 0


def test_mner_uses_single_span_without_pair_gate():
    model = make_model(task="mner")
    units, globals_, details = make_inputs(unit_count=9)
    del units["tail"], units["tail_width"]
    output = model(units, globals_, details, labels=torch.arange(9) % 4)
    assert model.pair_projection is None
    assert output["logits"].shape == (9, 4)
    assert output["gate"].shape == (9,)
    assert output["selected"].shape == (9, 1)
    assert output["query"].shape == (9, 8)
    torch.testing.assert_close(output["query"].norm(dim=-1), torch.ones(9))


@pytest.mark.parametrize("top_k,expected", [(1, [1]), (2, [1, 2])])
def test_stable_attachment_and_patch_ties_with_masks(top_k, expected):
    model = make_model(top_k=top_k).eval()
    units, _, _ = make_inputs()
    globals_ = torch.ones(4, 8)
    details = {index: torch.ones(7, 8) for index in range(4)}
    output = model(units, globals_, details, threshold=0,
                   global_mask=torch.tensor([False, True, True, True]))
    assert output["selected"].tolist() == [expected] * 3
    assert output["patchids"].tolist() == [[[0, 1, 2]] * top_k] * 3
    assert output["global_screen_calls"] == 3
    assert output["high_resolution_calls"] == top_k
    assert torch.isneginf(output["similarities"][:, 0]).all()


def test_masked_statistics_are_max_mean_population_std_top2_mean():
    model = make_model(query_dim=8).eval()
    with torch.no_grad():
        model.global_projection.weight.copy_(torch.eye(8))
        model.global_projection.bias.zero_()
        model.query_projection.weight.zero_()
        model.query_projection.bias.copy_(torch.tensor([1.0] + [0.0] * 7))
    units, _, _ = make_inputs()
    globals_ = torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7,
                             [0.0, 1.0] + [0.0] * 6, [100.0] * 8])
    output = model(units, globals_, global_mask=torch.tensor([True, True, True, False]), threshold=math.inf)
    expected = torch.tensor([1.0, 0.0, math.sqrt(2 / 3), 0.5]).expand(3, -1)
    torch.testing.assert_close(output["similarity_stats"], expected)


def test_single_valid_image_statistics_have_zero_std():
    model = make_model().eval()
    units, globals_, details = make_inputs(image_count=1)
    output = model(units, globals_, details, threshold=0)
    stats = output["similarity_stats"]
    assert torch.isfinite(stats).all()
    assert stats[:, 2].eq(0).all()
    assert torch.equal(stats[:, 0], stats[:, 1])
    assert torch.equal(stats[:, 0], stats[:, 3])


def test_provider_is_called_once_on_deduplicated_selected_indices():
    model = make_model(top_k=2).eval()
    units, _, details = make_inputs(unit_count=20)
    globals_ = torch.ones(3, 8)
    requests = []

    def provider(indices):
        requests.append(indices)
        return {index: details[index] for index in indices}

    output = model(units, globals_, detail_provider=provider, threshold=0)
    assert requests == [[0, 1]]
    assert output["high_resolution_calls"] == output["detail_cache_misses"] == 2
    assert output["active"].sum() == 20
    assert output["patchids"].shape == (20, 2, 3)


def test_cached_details_do_not_call_provider_and_missing_cache_subset_is_explicit():
    model = make_model(top_k=2).eval()
    units, _, details = make_inputs()
    globals_ = torch.ones(3, 8)
    requests = []

    def provider(indices):
        requests.append(indices)
        return [details[index] for index in indices]

    output = model(units, globals_, {0: details[0]}, detail_provider=provider, threshold=0)
    assert requests == [[1]]
    assert output["detail_cache_misses"] == 1
    cached = model(units, globals_, details, detail_provider=provider, threshold=0)
    assert requests == [[1]]
    assert cached["detail_cache_misses"] == 0
    assert cached["high_resolution_calls"] == 2
    torch.testing.assert_close(output["logits"], cached["logits"])


@pytest.mark.parametrize("format_", ["single", "batch", "sequence"])
def test_provider_tensor_and_sequence_return_formats(format_):
    model = make_model().eval()
    units, globals_, details = make_inputs(image_count=1)

    def provider(indices):
        if format_ == "single":
            return details[0]
        if format_ == "batch":
            return details[0][None]
        return [details[0]]

    output = model(units, globals_, detail_provider=provider, threshold=0)
    assert output["high_resolution_calls"] == 1
    assert output["vision_available"].all()


def test_maxsim_gate_has_no_parameterized_gate_or_rng_dependency():
    model = make_model().eval()
    units, globals_, details = make_inputs()

    def forbidden_gate(module, arguments):
        pytest.fail("MaxSim called the trainable gate MLP")

    handle = model.gate_mlp.register_forward_pre_hook(forbidden_gate)
    output = model(units, globals_, details, method="MaxSim+Top-1", threshold=0, labels=torch.tensor([0, 1, 2]))
    expected = (1 + output["similarities"].max(-1).values) / 2
    torch.testing.assert_close(output["gate"], expected)
    output["loss"].backward()
    assert gradient_sum(model.gate_mlp) == 0
    assert output["utility_loss"] == output["budget_loss"] == 0
    handle.remove()


def test_text_only_never_uses_visual_modules_even_with_supplied_details():
    model = make_model()
    units, globals_, details = make_inputs()

    def forbidden(module, arguments):
        pytest.fail("Text-only called a visual projection or learned gate")

    handles = [module.register_forward_pre_hook(forbidden) for module in
               (model.global_projection, model.patch_projection, model.gate_mlp)]
    output = model(units, globals_, details, method="text-only", force_details=True,
                   labels=torch.tensor([0, 1, 2]))
    assert torch.equal(output["logits"], output["text_logits"])
    assert output["high_resolution_calls"] == output["global_screen_calls"] == 0
    assert torch.equal(output["loss"], output["task_loss"])
    for handle in handles:
        handle.remove()


def test_always_vision_ignores_threshold_and_uses_all_samples():
    model = make_model().eval()
    units, globals_, details = make_inputs()
    output = model(units, globals_, details, method="always-vision", threshold=math.inf)
    assert output["active"].all()
    assert output["eta"].eq(1).all()
    assert output["high_resolution_calls"] == 3
    torch.testing.assert_close(output["logits"], output["vision_logits"])


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("top_k", [1, 2])
def test_always_vision_encodes_all_valid_attachments_but_pools_only_top1(training, top_k):
    model = make_model(top_k=top_k).eval()
    units, _, _ = make_inputs(unit_count=5)
    globals_ = torch.ones(4, 8)
    image_mask = torch.tensor([False, True, True, True])
    details = {index: torch.randn(256, 8) for index in range(1, 4)}
    requests = []

    def provider(indices):
        requests.append(indices)
        return {index: details[index] for index in indices}

    output = model(units, globals_, detail_provider=provider, method="always_vision",
                   training=training, global_mask=image_mask, threshold=math.inf)
    reference = model(units, globals_, details, method="saver", training=False,
                      global_mask=image_mask, threshold=0, top_k=1)
    assert requests == [[1, 2, 3]]
    assert output["high_resolution_calls"] == output["detail_cache_misses"] == 3
    assert output["global_screen_calls"] == 3
    assert output["selected"].tolist() == [[1]] * 5
    assert output["patchids"].shape == (5, 1, 3)
    assert torch.equal(output["selected"], reference["selected"])
    assert torch.equal(output["patchids"], reference["patchids"])
    torch.testing.assert_close(output["evidence"], reference["evidence"])
    torch.testing.assert_close(output["logits"], reference["logits"])


def test_always_vision_requests_uncached_unselected_attachments():
    model = make_model().eval()
    units, _, details = make_inputs()
    globals_ = torch.ones(3, 8)
    requests = []

    def provider(indices):
        requests.append(indices)
        return {index: details[index] for index in indices}

    output = model(units, globals_, {0: details[0]}, detail_provider=provider,
                   method="always_vision")
    assert requests == [[1, 2]]
    assert output["high_resolution_calls"] == 3
    assert output["detail_cache_misses"] == 2
    assert output["selected"].tolist() == [[0]] * 3


def test_random_gate_uses_frozen_matching_rate_and_external_draws():
    model = make_model().eval()
    units, globals_, details = make_inputs()
    draws = torch.tensor([0.1, 0.35, 0.8])
    output = model(units, globals_, details, method="random", random_draws=draws,
                   random_rate=0.35, threshold=math.inf)
    assert output["active"].tolist() == [True, False, False]
    assert output["gate"].eq(0.35).all()
    assert torch.equal(output["logits"][1:], output["text_logits"][1:])
    assert output["selected"][1:].eq(-1).all()
    with pytest.raises(ValueError):
        model(units, globals_, details, method="random")


def test_mean_pool_is_full_patch_mean_and_does_not_use_attention():
    model = make_model(pooling="mean").eval()
    units, globals_, details = make_inputs(image_count=1)

    def forbidden(module, arguments):
        pytest.fail("Mean-pooling ablation called query attention")

    handle = model.patch_attention.register_forward_pre_hook(forbidden)
    output = model(units, globals_, details, threshold=0)
    globals_projected = nn.functional.normalize(model.global_projection(globals_), dim=-1)
    patches_projected = nn.functional.normalize(model.patch_projection(details[0]), dim=-1)
    expected = torch.cat((globals_projected, patches_projected), 0).mean(0)
    torch.testing.assert_close(output["evidence"], expected[None].expand(3, -1))
    assert output["patchids"].tolist() == [[[0, 1, 2, 3, 4, 5, 6]]] * 3
    handle.remove()


def test_padded_units_and_invalid_images_excluded_with_exact_fallback():
    model = make_model()
    units, globals_, details = make_inputs()
    units["mask"] = torch.tensor([True, False, True])
    units["head"][1] = units["tail"][1] = math.nan
    units["head_width"][1] = units["tail_width"][1] = -1
    globals_[0] = math.nan
    output = model(units, globals_, details, global_mask=torch.tensor([False, True, True]),
                   labels=torch.tensor([0, -100, 2]))
    assert output["gate"][1] == output["eta"][1] == 0
    assert not output["active"][1]
    assert torch.equal(output["logits"][1], output["text_logits"][1])
    assert output["selected"][1].eq(-1).all()
    assert output["selected"].ne(0).all()
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["loss"])
    expected = (output["gate"][[0, 2]].mean() - model.rho).square()
    torch.testing.assert_close(output["budget_loss"], expected)


def test_all_invalid_images_skip_details_and_return_text():
    model = make_model()
    units, globals_, _ = make_inputs()
    output = model(units, globals_, global_mask=torch.zeros(3, dtype=torch.bool))
    assert torch.equal(output["logits"], output["text_logits"])
    assert not output["active"].any()
    assert output["high_resolution_calls"] == 0


def test_topk_larger_than_attachment_bank_is_padded():
    model = make_model(top_k=2).eval()
    units, globals_, details = make_inputs(image_count=1)
    output = model(units, globals_, details, threshold=0)
    assert output["selected"].tolist() == [[0, -1]] * 3
    assert output["high_resolution_calls"] == 1


def test_empty_unit_bank_has_finite_zero_loss():
    model = make_model()
    units, globals_, _ = make_inputs(unit_count=0)
    output = model(units, globals_, labels=torch.empty(0, dtype=torch.long))
    assert output["logits"].shape == (0, 4)
    assert output["high_resolution_calls"] == 0
    assert output["loss"] == 0


def test_missing_detail_features_and_bad_width_fail_loudly():
    model = make_model().eval()
    units, globals_, _ = make_inputs()
    with pytest.raises(ValueError, match="Missing high-resolution"):
        model(units, globals_, threshold=0)
    units["head_width"][0] = 0
    with pytest.raises(ValueError, match="outside"):
        model(units, globals_, threshold=math.inf)
