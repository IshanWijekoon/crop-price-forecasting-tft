"""Diebold-Mariano test of equal predictive accuracy (Diebold and Mariano, 1995).

With the small-sample correction of Harvey, Leybourne and Newbold (1997) and a Student-t
reference with T - 1 degrees of freedom. Forecasts made h steps ahead have loss
differentials that are serially correlated up to lag h - 1, so the long-run variance
uses h - 1 autocovariances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

logger = logging.getLogger(__name__)

ALTERNATIVES = ("two-sided", "less", "greater")
VARIANCE_ESTIMATORS = ("acf", "bartlett")


def _autocovariances(d: np.ndarray, max_lag: int) -> np.ndarray:
    centred = d - d.mean()
    n = len(d)
    return np.array([np.dot(centred[k:], centred[: n - k]) / n for k in range(max_lag + 1)])


def long_run_variance(d: np.ndarray, h: int, estimator: str = "acf") -> float:
    """`acf`: unweighted autocovariances up to h - 1, as in DM (1995). Can be negative.
    `bartlett`: Newey-West weights 1 - k/h, always non-negative."""
    gamma = _autocovariances(d, h - 1)
    if estimator == "acf":
        weights = np.ones(h - 1)
    elif estimator == "bartlett":
        weights = 1.0 - np.arange(1, h) / h
    else:
        raise ValueError(f"estimator must be one of {VARIANCE_ESTIMATORS}, got {estimator!r}")
    return float(gamma[0] + 2.0 * np.dot(weights, gamma[1:]))


def diebold_mariano_test(
    loss_a: ArrayLike,
    loss_b: ArrayLike,
    h: int = 1,
    alternative: str = "two-sided",
    variance: str = "acf",
) -> dict[str, Any]:
    """Test H0: E[loss_a - loss_b] = 0.

    `alternative="less"` means model A has lower expected loss. With `variance="acf"`, a
    non-positive variance estimate falls back to Bartlett weights; `variance_estimator`
    in the result says which was used.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"Loss series must be 1-D and the same length, got {a.shape} and {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Loss series must be finite")
    n = len(a)
    if h < 1 or h >= n:
        raise ValueError(f"Need 1 <= h < number of losses, got h={h}, n={n}")
    hln_ratio = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if hln_ratio <= 0:
        raise ValueError(f"Too few losses (n={n}) for the small-sample correction at h={h}")

    d = a - b
    estimator = variance
    lrv = long_run_variance(d, h, estimator)
    if lrv <= 0 and estimator == "acf":
        estimator = "bartlett"
        lrv = long_run_variance(d, h, estimator)

    result: dict[str, Any] = {
        "n": n,
        "h": h,
        "mean_loss_a": float(a.mean()),
        "mean_loss_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "variance_estimator": estimator,
        "alternative": alternative,
    }
    if lrv <= 0:
        logger.warning("DM test: loss differential has zero variance; statistic undefined")
        return {**result, "dm_stat": float("nan"), "hln_stat": float("nan"), "p_value": float("nan")}

    dm_stat = float(d.mean() / np.sqrt(lrv / n))
    hln_stat = float(np.sqrt(hln_ratio) * dm_stat)
    dist = stats.t(df=n - 1)
    if alternative == "two-sided":
        p_value = 2.0 * dist.sf(abs(hln_stat))
    elif alternative == "less":
        p_value = dist.cdf(hln_stat)
    else:
        p_value = dist.sf(hln_stat)
    return {**result, "dm_stat": dm_stat, "hln_stat": hln_stat, "p_value": float(p_value)}
