from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_temporal_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    # TODO: Add day/week/month indicators and optional cyclical encodings.
    logger.debug(
        "generate_temporal_features called (stub) date_column=%s",
        date_column,
    )
    raise NotImplementedError("generate_temporal_features is not implemented yet.")
