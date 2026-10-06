from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def synchronize_time_series(
    df: pd.DataFrame,
    date_column: str = "date",
    freq: str = "D",
    group_cols: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reindex every group onto one shared calendar. New rows are left missing."""
    start = pd.Timestamp(start) if start is not None else df[date_column].min()
    end = pd.Timestamp(end) if end is not None else df[date_column].max()
    calendar = pd.date_range(start, end, freq=freq, name=date_column)

    if not group_cols:
        out = df.set_index(date_column).reindex(calendar).reset_index()
        return out

    group_cols = list(group_cols)
    groups = df[group_cols].drop_duplicates()
    grid = groups.merge(pd.DataFrame({date_column: calendar}), how="cross")
    window = df[(df[date_column] >= start) & (df[date_column] <= end)]
    out = grid.merge(window, on=[*group_cols, date_column], how="left", validate="one_to_one")
    logger.debug("synchronize_time_series: %d groups x %d dates", len(groups), len(calendar))
    return out.sort_values([*group_cols, date_column]).reset_index(drop=True)


def resample_to_daily(
    df: pd.DataFrame,
    date_column: str = "date",
    aggregation: str = "mean",
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Collapse several reports on the same day into one row per group and day."""
    keys = [*(group_cols or []), pd.Grouper(key=date_column, freq="D")]
    numeric = df.select_dtypes("number").columns.difference(list(group_cols or []))
    out = df.groupby(keys)[list(numeric)].agg(aggregation).reset_index()
    return out.dropna(subset=list(numeric), how="all")
