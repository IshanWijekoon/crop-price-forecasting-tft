from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def generate_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
    stats: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    # TODO: Compute causal rolling stats aligned to each forecast origin.
    logger.debug(
        "generate_rolling_features called (stub) windows=%s stats=%s",
        windows,
        stats,
    )
    raise NotImplementedError("generate_rolling_features is not implemented yet.")
