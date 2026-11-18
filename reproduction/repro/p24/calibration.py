"""Finite-grid forced-vision error audit from SAVER Section 3.

Calibration inputs must come from a frozen checkpoint on held-out data.
These bounds concern forced-vision classification error on the named audit
population, not task F1, routed error, or the visual-harm indicator.
"""

import math
import operator
from collections.abc import Iterable

import numpy as np
from scipy.stats import beta


THRESHOLD_GRID = tuple(index / 100 for index in range(101))


def cp_upper(k: int, n: int, confidence: float = 1 - 0.05 / 101) -> float:
    """Exact one-sided Clopper--Pearson upper limit, with n=0 infeasible."""
    try:
        k, n = operator.index(k), operator.index(n)
    except TypeError as error:
        raise ValueError("k and n must be integer counts") from error
    if n < 0 or not 0 <= k <= n:
        raise ValueError("Counts must satisfy 0 <= k <= n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if n == 0 or k == n:
        return 1.0
    return float(beta.ppf(confidence, k + 1, n - k))


clopper_pearson_upper = cp_upper


def _array(values: Iterable, name: str) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = list(values)
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def calibrate_threshold(
    gates: Iterable,
    forced_errors: Iterable,
    *,
    alpha: float = 0.10,
    delta: float = 0.05,
    thresholds: Iterable | None = None,
    mask: Iterable | None = None,
) -> dict:
    """Select max coverage subject to CP <= alpha; ties use smallest t.

    Scores are gate probabilities and errors are exactly 0/1 forced-vision
    misclassification indicators. ``mask`` excludes padded/invalid-image
    units from the explicitly chosen calibration population. The default grid
    has 101 thresholds and uses simultaneous confidence 1-delta/101. Empty
    subsets are never feasible. Failure returns ``threshold=math.inf`` and
    all-off coverage; alpha and the gate are never relaxed.

    The result and each curve row contain native Python scalars. Infinity
    should be encoded explicitly by the runner when writing strict JSON.
    """
    if not 0 <= alpha <= 1 or not 0 < delta < 1:
        raise ValueError("alpha must be in [0,1] and delta strictly in (0,1)")
    scores = _array(gates, "gates").astype(float)
    errors = _array(forced_errors, "forced_errors")
    if scores.shape != errors.shape:
        raise ValueError("gates and forced_errors must have identical shapes")
    if mask is not None:
        valid = _array(mask, "mask")
        if valid.shape != scores.shape or not np.isin(valid, [0, 1]).all():
            raise ValueError("mask must be a boolean vector matching gates")
        scores, errors = scores[valid.astype(bool)], errors[valid.astype(bool)]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("gates must be finite values in [0,1]")
    if not np.isin(errors, [0, 1]).all():
        raise ValueError("forced_errors must be binary classification errors")
    grid = np.asarray(THRESHOLD_GRID if thresholds is None else _array(thresholds, "thresholds"), dtype=float)
    if not grid.size or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("thresholds must be a nonempty finite grid in [0,1]")
    grid = np.unique(grid)
    confidence = 1 - delta / len(grid)
    curve, best = [], None
    for threshold in grid:
        active = scores >= threshold
        n = int(active.sum())
        k = int(errors[active].sum())
        upper = cp_upper(k, n, confidence)
        row = {"threshold": float(threshold), "n": n, "k": k,
               "coverage": n / len(scores) if len(scores) else 0.0,
               "risk": k / n if n else None, "upper": upper,
               "feasible": n > 0 and upper <= alpha}
        curve.append(row)
        if row["feasible"] and (best is None or n > best["n"]):
            best = row
    result = {"threshold": math.inf, "feasible": False, "coverage": 0.0,
              "n": 0, "k": 0, "risk": None, "upper": 1.0}
    if best is not None:
        result.update(best)
    result.update({"alpha": float(alpha), "delta": float(delta), "confidence": float(confidence),
                   "grid_size": len(grid), "n_total": len(scores), "curve": curve,
                   "error_definition": "forced_vision_misclassification"})
    return result


calibrate = calibrate_threshold
