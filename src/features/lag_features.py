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
    # TODO: Create lag_{column}_{k} features without leaking future information.
    logger.debug(
        "generate_lag_features called (stub) columns=%s lags=%s",
        columns,
        lags,
    )
    raise NotImplementedError("generate_lag_features is not implemented yet.")
