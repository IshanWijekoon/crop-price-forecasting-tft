from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_forecast_windows(
    frame: pd.DataFrame,
    encoder_length: int,
    horizon: int,
    step: int = 1,
    date_column: str = "date",
    group_column: str = "crop",
    split_column: str = "split",
    observed_column: str = "price_observed",
) -> pd.DataFrame:
    """Cut sliding encoder/decoder windows, one row per group and forecast origin.

    The origin is the last encoder day; the decoder is the next `horizon` days. A window
    belongs to the split that holds its whole decoder, and windows whose decoder crosses
    a split boundary are dropped. The encoder may reach into earlier splits because those
    days are already known at the origin.
    """
    if encoder_length < 1 or horizon < 1 or step < 1:
        raise ValueError("encoder_length, horizon, and step must all be >= 1")

    rows = []
    for group, g in frame.groupby(group_column, sort=True):
        g = g.sort_values(date_column)
        dates = pd.DatetimeIndex(g[date_column])
        if len(dates) > 1 and not (np.diff(dates.values) == np.timedelta64(1, "D")).all():
            raise ValueError(f"{group_column}={group!r} is not on a gap-free daily calendar")

        splits = g[split_column].to_numpy()
        observed = g[observed_column].to_numpy()
        for i in range(encoder_length - 1, len(g) - horizon, step):
            decoder = slice(i + 1, i + 1 + horizon)
            decoder_splits = np.unique(splits[decoder])
            if len(decoder_splits) != 1:
                continue
            rows.append(
                {
                    group_column: group,
                    "encoder_start": dates[i - encoder_length + 1],
                    "origin": dates[i],
                    "decoder_start": dates[i + 1],
                    "decoder_end": dates[i + horizon],
                    "split": decoder_splits[0],
                    "observed_targets": int(observed[decoder].sum()),
                }
            )

    windows = pd.DataFrame(rows)
    if windows.empty:
        raise ValueError(
            f"No complete windows for encoder_length={encoder_length}, horizon={horizon}"
        )
    logger.info(
        "Forecast windows per split: %s",
        windows.groupby("split").size().to_dict(),
    )
    return windows


def slice_window(
    frame: pd.DataFrame,
    window: pd.Series,
    date_column: str = "date",
    group_column: str = "crop",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the (encoder, decoder) rows of `frame` for one row of `make_forecast_windows`."""
    g = frame[frame[group_column].eq(window[group_column])]
    encoder = g[g[date_column].between(window["encoder_start"], window["origin"])]
    decoder = g[g[date_column].between(window["decoder_start"], window["decoder_end"])]
    return encoder, decoder
