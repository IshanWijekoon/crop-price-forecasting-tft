from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def synchronize_time_series(
    df: pd.DataFrame,
    date_column: str = "date",
    freq: str = "D",
) -> pd.DataFrame:
    # TODO: Reindex to a complete calendar and align multi-crop series.
    logger.debug(
        "synchronize_time_series called (stub) date_column=%s freq=%s",
        date_column,
        freq,
    )
    raise NotImplementedError("synchronize_time_series is not implemented yet.")


def resample_to_daily(
    df: pd.DataFrame,
    date_column: str = "date",
    aggregation: str = "mean",
) -> pd.DataFrame:
    # TODO: Aggregate intra-day or multi-report days into a single daily value.
    logger.debug("resample_to_daily called (stub) aggregation=%s", aggregation)
    raise NotImplementedError("resample_to_daily is not implemented yet.")
