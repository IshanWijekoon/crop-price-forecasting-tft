from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

SCALERS = {"minmax": MinMaxScaler, "standard": StandardScaler}


def normalize_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "minmax",
    fitted_scaler: Any | None = None,
    fit_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Scale `columns`. Fit only on rows where `fit_mask` is True (the training split).

    Pass the returned scaler back as `fitted_scaler` to transform validation and test
    rows without refitting.
    """
    columns = list(columns)
    out = df.copy()

    if fitted_scaler is None:
        if method not in SCALERS:
            raise ValueError(f"Unknown scaling method {method!r}; expected {list(SCALERS)}")
        if fit_mask is None:
            raise ValueError("fit_mask is required when fitting, so the scaler only sees training rows")
        fitted_scaler = SCALERS[method]()
        fitted_scaler.fit(out.loc[fit_mask, columns])
        logger.debug("Fitted %s scaler on %d rows", method, int(fit_mask.sum()))

    out[columns] = fitted_scaler.transform(out[columns])
    return out, fitted_scaler


def scaler_to_dict(scaler: Any, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-column fitted parameters, for saving next to the scaled data."""
    if isinstance(scaler, MinMaxScaler):
        params = {"min": scaler.data_min_, "max": scaler.data_max_}
    elif isinstance(scaler, StandardScaler):
        params = {"mean": scaler.mean_, "std": scaler.scale_}
    else:
        raise TypeError(f"Unsupported scaler {type(scaler).__name__}")
    return {c: {k: float(v[i]) for k, v in params.items()} for i, c in enumerate(columns)}
