from __future__ import annotations

import logging
from functools import reduce
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(
    frames: Sequence[pd.DataFrame],
    on: Sequence[str] | None = None,
    how: str = "outer",
) -> pd.DataFrame:
    if not frames:
        raise ValueError("merge_datasets needs at least one frame")
    on = list(on or ["date"])

    for i, frame in enumerate(frames):
        missing = [c for c in on if c not in frame.columns]
        if missing:
            raise KeyError(f"Frame {i} is missing merge keys {missing}")
        if frame.duplicated(on).any():
            raise ValueError(f"Frame {i} has duplicate rows for keys {on}")

    merged = reduce(
        lambda left, right: left.merge(right, on=on, how=how, validate="one_to_one"),
        frames,
    )
    return merged.sort_values(on).reset_index(drop=True)


def align_on_date(
    left: pd.DataFrame,
    right: pd.DataFrame,
    date_column: str = "date",
    how: str = "left",
    on: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Join covariates onto a panel by date (plus any extra keys), one right row per key."""
    keys = [date_column, *(on or [])]
    left = left.copy()
    right = right.copy()
    left[date_column] = pd.to_datetime(left[date_column]).dt.normalize()
    right[date_column] = pd.to_datetime(right[date_column]).dt.normalize()

    unmatched = ~left[date_column].isin(right[date_column])
    if unmatched.any():
        logger.warning("%d left rows have no matching %s on the right", int(unmatched.sum()), date_column)

    return left.merge(right, on=keys, how=how, validate="many_to_one")
