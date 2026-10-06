from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def generate_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add `{column}_lag_{k}` = value k rows earlier in the same group.

    Rows must already be sorted by date inside each group on a regular calendar,
    so a row shift equals a time shift.
    """
    if any(k < 1 for k in lags):
        raise ValueError("Lags must be >= 1 to avoid using same-period values")

    out = df.copy()
    grouped = out.groupby(list(group_cols), sort=False) if group_cols else None
    for column in columns:
        source = grouped[column] if grouped is not None else out[column]
        for k in lags:
            out[f"{column}_lag_{k}"] = source.shift(k)
    logger.debug("generate_lag_features: columns=%s lags=%s", columns, lags)
    return out
