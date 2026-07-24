from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(
    frames: Sequence[pd.DataFrame],
    on: Sequence[str] | None = None,
    how: str = "outer",
) -> pd.DataFrame:
    # TODO: Validate schemas and merge price series with external covariates.
    logger.debug("merge_datasets called (stub) with how=%s", how)
    raise NotImplementedError("merge_datasets is not implemented yet.")


def align_on_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    # TODO: Ensure both sides share a compatible date index before joining.
    logger.debug("align_on_date called (stub) for column=%s", date_column)
    raise NotImplementedError("align_on_date is not implemented yet.")
