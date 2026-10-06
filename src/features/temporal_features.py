from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sri Lanka Department of Meteorology monsoon calendar.
MONSOON_SEASON_BY_MONTH = {
    12: "northeast_monsoon",
    1: "northeast_monsoon",
    2: "northeast_monsoon",
    3: "first_intermonsoon",
    4: "first_intermonsoon",
    5: "southwest_monsoon",
    6: "southwest_monsoon",
    7: "southwest_monsoon",
    8: "southwest_monsoon",
    9: "southwest_monsoon",
    10: "second_intermonsoon",
    11: "second_intermonsoon",
}


def monsoon_season(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.month.map(MONSOON_SEASON_BY_MONTH)


def generate_temporal_features(
    df: pd.DataFrame,
    date_column: str = "date",
) -> pd.DataFrame:
    """Calendar features that are known in advance for any future date."""
    out = df.copy()
    dates = pd.to_datetime(out[date_column])
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_year"] = dates.dt.dayofyear
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int64")
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["season"] = monsoon_season(dates)
    return out
