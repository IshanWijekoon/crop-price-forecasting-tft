from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Orchestrate duplicate removal, date standardization, and column checks.
    logger.debug("clean_data called (stub)")
    raise NotImplementedError("clean_data is not implemented yet.")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Define duplicate keys (e.g., date, market, crop) and drop strategy.
    logger.debug("remove_duplicates called (stub)")
    raise NotImplementedError("remove_duplicates is not implemented yet.")


def standardize_dates(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    # TODO: Parse dates, sort chronologically, and enforce a daily frequency policy.
    logger.debug("standardize_dates called (stub) for column=%s", date_column)
    raise NotImplementedError("standardize_dates is not implemented yet.")
