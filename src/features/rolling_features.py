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
    min_periods: dict[int, int] | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Add `{name or column}_roll_{stat}_{w}` over the previous w rows, excluding the current row.

    Missing values inside the window are skipped, so a window over a calendar grid
    with closed-market days uses only the days that were actually observed.
    """
    stats = list(stats or ["mean", "std"])
    min_periods = min_periods or {}
    out = df.copy()

    for column in columns:
        prefix = name or column
        for w in windows:
            mp = min_periods.get(w, max(1, w // 3))
            for stat in stats:

                def _roll(s: pd.Series, w: int = w, mp: int = mp, stat: str = stat) -> pd.Series:
                    return s.shift(1).rolling(w, min_periods=mp).agg(stat)

                if group_cols:
                    rolled = out.groupby(list(group_cols), sort=False)[column].transform(_roll)
                else:
                    rolled = _roll(out[column])
                out[f"{prefix}_roll_{stat}_{w}"] = rolled

    logger.debug("generate_rolling_features: windows=%s stats=%s", windows, stats)
    return out


def generate_observation_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: Sequence[int],
    stats: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Add `{name}_roll_{stat}_{w}q` over the last w observed values strictly before each row.

    Windows count actual observations, not rows, so long gaps (market holidays) do not
    empty the window. Rows must be on a regular calendar, sorted by date inside each group.
    """
    stats = list(stats or ["mean", "std"])
    prefix = name or column
    out = df.copy()

    def _roll(s: pd.Series, w: int, stat: str) -> pd.Series:
        observed = s.dropna()
        inclusive = observed.rolling(w, min_periods=w).agg(stat)
        return inclusive.reindex(s.index).ffill().shift(1)

    for w in windows:
        for stat in stats:
            if group_cols:
                rolled = out.groupby(list(group_cols), sort=False)[column].transform(
                    lambda s, w=w, stat=stat: _roll(s, w, stat)
                )
            else:
                rolled = _roll(out[column], w, stat)
            out[f"{prefix}_roll_{stat}_{w}q"] = rolled
    return out
