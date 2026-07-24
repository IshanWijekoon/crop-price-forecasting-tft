from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "ffill",
) -> pd.DataFrame:
    # TODO: Support forward-fill, interpolation, and crop-specific policies.
    logger.debug("handle_missing_values called (stub) with strategy=%s", strategy)
    raise NotImplementedError("handle_missing_values is not implemented yet.")


def impute_series(
    series: pd.Series,
    method: str = "interpolate",
) -> pd.Series:
    # TODO: Implement time-aware interpolation suitable for daily prices.
    logger.debug("impute_series called (stub) with method=%s", method)
    raise NotImplementedError("impute_series is not implemented yet.")
