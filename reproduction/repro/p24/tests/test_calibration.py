import math

import numpy as np
import pytest
from scipy.stats import beta, binom

from repro.p24.calibration import (
    THRESHOLD_GRID,
    calibrate_threshold,
    clopper_pearson_upper,
    cp_upper,
)


def test_zero_errors_upper_matches_closed_form_and_binomial_tail():
    confidence = 1 - 0.05 / 101
    upper = cp_upper(0, 100, confidence)
    assert upper == pytest.approx(1 - (1 - confidence) ** (1 / 100))
    assert binom.cdf(0, 100, upper) == pytest.approx(1 - confidence)


@pytest.mark.parametrize("errors,total", [(1, 10), (10, 258), (161, 2116), (665, 9448)])
def test_exact_cp_matches_beta_quantile_and_binomial_tail(errors, total):
    confidence = 1 - 0.05 / 101
    upper = cp_upper(errors, total, confidence)
    assert upper == pytest.approx(beta.ppf(confidence, errors + 1, total - errors))
    assert binom.cdf(errors, total, upper) == pytest.approx(1 - confidence)
    assert upper > errors / total
    assert clopper_pearson_upper(errors, total, confidence) == upper


def test_empty_and_all_error_counts_are_upper_one():
    assert cp_upper(0, 0) == 1
    assert cp_upper(10, 10) == 1


@pytest.mark.parametrize("errors,total", [(-1, 1), (2, 1), (1, -1), (0.5, 1), (0, 1.5)])
def test_invalid_counts_raise(errors, total):
    with pytest.raises(ValueError):
        cp_upper(errors, total)


@pytest.mark.parametrize("confidence", [0, 1, -1, math.nan])
def test_invalid_confidence_raises(confidence):
    with pytest.raises(ValueError):
        cp_upper(0, 10, confidence)


def test_grid_is_exact_101_point_prespecified_family():
    assert len(THRESHOLD_GRID) == 101
    assert THRESHOLD_GRID == tuple(index / 100 for index in range(101))


def test_feasible_full_coverage_chooses_smallest_threshold_on_ties():
    result = calibrate_threshold(np.full(200, 0.8), np.zeros(200))
    assert result["feasible"]
    assert result["threshold"] == 0
    assert result["coverage"] == 1
    assert result["n"] == result["n_total"] == 200
    assert result["k"] == 0
    assert result["upper"] <= 0.1
    assert result["confidence"] == 1 - 0.05 / 101
    assert result["grid_size"] == len(result["curve"]) == 101
    assert result["error_definition"] == "forced_vision_misclassification"


def test_largest_feasible_coverage_ignores_lowest_risk_alternative():
    scores = np.concatenate([np.full(100, 0.2), np.full(500, 0.8), np.full(200, 0.95)])
    errors = np.concatenate([np.ones(100), np.r_[np.ones(10), np.zeros(490)], np.zeros(200)])
    result = calibrate_threshold(scores, errors)
    assert result["feasible"]
    assert result["threshold"] == 0.21
    assert result["n"] == 700
    assert result["k"] == 10
    assert result["coverage"] == 700 / 800
    assert result["risk"] == 10 / 700
    assert not result["curve"][20]["feasible"]
    assert result["curve"][21]["feasible"]


def test_threshold_is_inclusive_and_ties_ignore_input_grid_order():
    result = calibrate_threshold([0.8] * 200, [0] * 200, thresholds=[1, 0.8, 0.5, 0.8])
    assert result["threshold"] == 0.5
    assert result["grid_size"] == 3
    assert result["confidence"] == 1 - 0.05 / 3
    assert [row["n"] for row in result["curve"]] == [200, 200, 0]


@pytest.mark.parametrize("scores,errors", [([], []), ([0.9] * 100, [1] * 100), ([0.5] * 10, [0] * 10)])
def test_no_feasible_nonempty_subset_is_infinity_all_off(scores, errors):
    result = calibrate_threshold(scores, errors)
    assert not result["feasible"]
    assert result["threshold"] == math.inf
    assert result["coverage"] == 0
    assert result["n"] == result["k"] == 0
    assert result["upper"] == 1
    assert result["risk"] is None
    assert not any(row["feasible"] for row in result["curve"])
    assert all(score < result["threshold"] for score in scores)


def test_empty_subsets_never_become_feasible_even_at_alpha_one():
    result = calibrate_threshold([], [], alpha=1)
    assert not result["feasible"]
    assert all(row["n"] == 0 and not row["feasible"] for row in result["curve"])


def test_explicit_population_mask_excludes_padded_or_invalid_image_units():
    scores = np.r_[np.full(200, 0.9), np.nan]
    errors = np.r_[np.zeros(200), np.nan]
    result = calibrate_threshold(scores, errors, mask=np.r_[np.ones(200), 0])
    assert result["n_total"] == result["n"] == 200
    assert result["feasible"]


def test_generator_inputs_and_numpy_integer_counts():
    result = calibrate_threshold((0.9 for _ in range(200)), (0 for _ in range(200)))
    assert result["coverage"] == 1
    assert cp_upper(np.int64(0), np.int64(200)) < 0.1


def test_cp_tightens_with_sample_size_and_increases_with_errors_or_confidence():
    assert cp_upper(0, 100) > cp_upper(0, 200)
    assert cp_upper(0, 200) < cp_upper(1, 200) < cp_upper(2, 200)
    assert cp_upper(1, 200, 0.95) < cp_upper(1, 200)


@pytest.mark.parametrize("kwargs", [
    {"gates": [1.1], "forced_errors": [0]},
    {"gates": [-0.1], "forced_errors": [0]},
    {"gates": [math.nan], "forced_errors": [0]},
    {"gates": [0.5], "forced_errors": [0.1]},
    {"gates": [0.5], "forced_errors": [0, 1]},
    {"gates": [[0.5]], "forced_errors": [0]},
    {"gates": [0.5], "forced_errors": [0], "alpha": -0.1},
    {"gates": [0.5], "forced_errors": [0], "delta": 0},
    {"gates": [0.5], "forced_errors": [0], "thresholds": []},
    {"gates": [0.5], "forced_errors": [0], "thresholds": [math.inf]},
    {"gates": [0.5], "forced_errors": [0], "mask": [True, False]},
])
def test_invalid_calibration_inputs_fail_without_relaxing_risk(kwargs):
    with pytest.raises(ValueError):
        calibrate_threshold(**kwargs)
