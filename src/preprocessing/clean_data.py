from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(
    df: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str],
    date_column: str = "date",
    date_format: str | None = None,
) -> pd.DataFrame:
    out = standardize_dates(df, date_column=date_column, date_format=date_format)
    out = remove_duplicates(out, keys=keys, value_columns=value_columns)

    for column in value_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column] <= 0
        if invalid.any():
            logger.warning(
                "%d non-positive values in %s set to missing", int(invalid.sum()), column
            )
            out.loc[invalid, column] = np.nan

    return out.reset_index(drop=True)


def remove_duplicates(
    df: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Drop exact duplicate rows, then average conflicting reports that share the same keys."""
    keys = list(keys)
    before = len(df)
    out = df.drop_duplicates()
    exact = before - len(out)

    conflicting = out.duplicated(keys, keep=False)
    if conflicting.any():
        value_columns = list(value_columns or out.select_dtypes("number").columns.difference(keys))
        other = [c for c in out.columns if c not in keys and c not in value_columns]
        agg = {c: "mean" for c in value_columns} | {c: "first" for c in other}
        out = out.groupby(keys, as_index=False, sort=False, dropna=False).agg(agg)

    logger.info(
        "remove_duplicates: %d exact duplicates dropped, %d rows had conflicting keys",
        exact,
        int(conflicting.sum()),
    )
    return out


def standardize_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    date_format: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out[date_column] = pd.to_datetime(out[date_column], format=date_format, errors="coerce")
    bad = out[date_column].isna()
    if bad.any():
        logger.warning("Dropping %d rows with unparseable %s", int(bad.sum()), date_column)
        out = out.loc[~bad]
    out[date_column] = out[date_column].dt.normalize()
    return out.sort_values(date_column, kind="stable").reset_index(drop=True)
