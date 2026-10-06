"""Forecast table shared by both models, so evaluation reads one schema.

One row per model, crop, forecast origin, and target day:
  model, crop, split, origin, date, horizon_day, price_observed, y_true, q10, q50, q90
`y_true` is empty on days without a quote; only rows with price_observed == 1 are scored.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["model", "crop", "split", "origin", "date", "horizon_day"]


def quantile_columns(quantiles: Sequence[float]) -> list[str]:
    return [f"q{round(q * 100):02d}" for q in quantiles]


def forecast_rows(
    model: str,
    crop: str,
    split: str,
    origin: pd.Timestamp,
    decoder: pd.DataFrame,
    quantile_values: np.ndarray,
    quantiles: Sequence[float],
    target: str,
    observed_flag: str,
) -> pd.DataFrame:
    """Rows for one forecast. `quantile_values` has shape (horizon, len(quantiles))."""
    if quantile_values.shape != (len(decoder), len(quantiles)):
        raise ValueError(
            f"Expected quantiles of shape {(len(decoder), len(quantiles))}, got {quantile_values.shape}"
        )
    out = pd.DataFrame(
        {
            "model": model,
            "crop": crop,
            "split": split,
            "origin": origin,
            "date": decoder["date"].to_numpy(),
            "horizon_day": (decoder["date"] - origin).dt.days.to_numpy(),
            "price_observed": decoder[observed_flag].to_numpy(),
            "y_true": decoder[target].to_numpy(),
        }
    )
    out[quantile_columns(quantiles)] = quantile_values
    return out


def finalize_forecasts(frames: list[pd.DataFrame], quantiles: Sequence[float]) -> pd.DataFrame:
    """Concatenate, sort quantiles that cross, and check one row per key."""
    forecasts = pd.concat(frames, ignore_index=True)
    qcols = quantile_columns(quantiles)
    values = forecasts[qcols].to_numpy()
    crossed = int((np.diff(values, axis=1) < 0).any(axis=1).sum())
    if crossed:
        logger.info("Sorted %d of %d rows with crossing quantiles", crossed, len(forecasts))
        forecasts[qcols] = np.sort(values, axis=1)
    if forecasts.duplicated(KEY_COLUMNS).any():
        raise ValueError("Forecast table has duplicate model/crop/origin/date rows")
    return forecasts.sort_values(KEY_COLUMNS).reset_index(drop=True)
