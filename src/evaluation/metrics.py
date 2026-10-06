"""Point and probabilistic forecast metrics. Inputs are 1-D arrays of scored rows only."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("No rows to score")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Metrics need finite values; filter unscored rows first")
    return y_true, y_pred


def absolute_errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    y_true, y_pred = _pair(y_true, y_pred)
    return np.abs(y_true - y_pred)


def pinball_losses(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> np.ndarray:
    """Per-row pinball loss: q * (y - f) if y >= f, else (1 - q) * (f - y)."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    y_true, y_pred = _pair(y_true, y_pred)
    diff = y_true - y_pred
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(absolute_errors(y_true, y_pred).mean())


def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error in percent. Rows with a zero target are left out."""
    y_true, y_pred = _pair(y_true, y_pred)
    nonzero = y_true != 0
    if not nonzero.any():
        raise ValueError("MAPE is undefined when every target is zero")
    if not nonzero.all():
        logger.warning("MAPE: left out %d rows with a zero target", int((~nonzero).sum()))
    return float(100.0 * np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))


def calculate_quantile_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    return float(pinball_losses(y_true, y_pred, quantile).mean())


def calculate_picp(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Share of targets inside the closed interval [lower, upper]."""
    y_true, lower = _pair(y_true, lower)
    _, upper = _pair(y_true, upper)
    if (lower > upper).any():
        raise ValueError("Interval lower bound exceeds upper bound")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def calculate_interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    lower, upper = _pair(lower, upper)
    return float(np.mean(upper - lower))
