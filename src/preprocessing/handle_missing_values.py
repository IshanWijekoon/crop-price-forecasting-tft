from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

STRATEGIES = ("ffill", "interpolate", "interpolate_ffill")


def handle_missing_values(
    df: pd.DataFrame,
    columns: Sequence[str],
    strategy: str = "ffill",
    date_column: str = "date",
    group_cols: Sequence[str] | None = None,
    flag_suffix: str | None = "_imputed",
) -> pd.DataFrame:
    """Impute covariates per group along the date axis.

    Only use this on predictors. Target prices are never imputed here.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    out = df.sort_values([*(group_cols or []), date_column]).copy()
    method = "ffill" if strategy == "ffill" else "interpolate"

    def _fill(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.set_index(date_column)
        for column in columns:
            filled = impute_series(frame[column], method=method)
            if strategy == "interpolate_ffill":
                filled = filled.ffill().bfill()
            frame[column] = filled
        return frame.reset_index()

    if flag_suffix:
        for column in columns:
            out[f"{column}{flag_suffix}"] = out[column].isna().astype("int8")

    if group_cols:
        parts = [_fill(g) for _, g in out.groupby(list(group_cols), sort=False)]
        out = pd.concat(parts, ignore_index=True)
    else:
        out = _fill(out)

    for column in columns:
        remaining = int(out[column].isna().sum())
        if remaining:
            logger.warning("%s still has %d missing values after %s", column, remaining, strategy)
    return out


def impute_series(
    series: pd.Series,
    method: str = "interpolate",
    limit: int | None = None,
) -> pd.Series:
    if method == "ffill":
        return series.ffill(limit=limit)
    if method == "interpolate":
        if isinstance(series.index, pd.DatetimeIndex):
            return series.interpolate(method="time", limit=limit, limit_area="inside")
        return series.interpolate(method="linear", limit=limit, limit_area="inside")
    raise ValueError(f"Unknown imputation method {method!r}")
