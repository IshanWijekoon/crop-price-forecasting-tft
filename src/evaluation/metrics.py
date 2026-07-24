from __future__ import annotations

import logging

from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    # TODO: Implement MAE = mean(|y_true - y_pred|).
    logger.debug("calculate_mae called (stub)")
    raise NotImplementedError("calculate_mae is not implemented yet.")


def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    # TODO: Implement RMSE = sqrt(mean((y_true - y_pred)^2)).
    logger.debug("calculate_rmse called (stub)")
    raise NotImplementedError("calculate_rmse is not implemented yet.")


def calculate_mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    # TODO: Guard against zero targets and implement MAPE.
    logger.debug("calculate_mape called (stub)")
    raise NotImplementedError("calculate_mape is not implemented yet.")


def calculate_quantile_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantile: float,
) -> float:
    # TODO: Implement pinball loss for probabilistic evaluation.
    logger.debug("calculate_quantile_loss called (stub) quantile=%s", quantile)
    raise NotImplementedError("calculate_quantile_loss is not implemented yet.")


def calculate_picp(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    # TODO: Compute coverage rate for prediction intervals.
    logger.debug("calculate_picp called (stub)")
    raise NotImplementedError("calculate_picp is not implemented yet.")
