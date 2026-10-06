"""SARIMAX baseline: one model per crop on the daily calendar.

Days without a quote stay missing in the endogenous series; the Kalman filter skips them.
Parameters are estimated on training rows only. At each forecast origin the fitted
parameters are re-applied to the history up to that origin, then projected `horizon` days.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.features.prepare_model_data import ModelData
from src.models.forecast_output import finalize_forecasts, forecast_rows

logger = logging.getLogger(__name__)

MODEL_NAME = "sarimax"


@dataclass(frozen=True)
class SarimaxSpec:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str | None


class SarimaxForecaster:
    def __init__(self, spec: SarimaxSpec, log_target: bool = True, maxiter: int = 200):
        self.spec = spec
        self.log_target = log_target
        self.maxiter = maxiter
        self.result = None

    def fit(self, price: pd.Series, exog: pd.DataFrame) -> "SarimaxForecaster":
        self.result = _fit(to_endog(price, self.log_target), exog, self.spec, self.maxiter)
        return self

    def forecast(
        self,
        price_history: pd.Series,
        exog_history: pd.DataFrame,
        exog_future: pd.DataFrame,
        quantiles: Sequence[float],
    ) -> np.ndarray:
        """Quantiles of shape (len(exog_future), len(quantiles)), in price units.

        Gaussian on the modelling scale; with a log target this gives log-normal price quantiles.
        """
        if self.result is None:
            raise RuntimeError("fit() must be called before forecast()")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered = self.result.apply(to_endog(price_history, self.log_target), exog=exog_history)
            fc = filtered.get_forecast(steps=len(exog_future), exog=exog_future)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        se = np.asarray(fc.se_mean, dtype=float)
        values = mean[:, None] + se[:, None] * norm.ppf(np.asarray(quantiles))[None, :]
        return np.exp(values) if self.log_target else values


def to_endog(price: pd.Series, log_target: bool) -> pd.Series:
    return np.log(price) if log_target else price


def _fit(endog: pd.Series, exog: pd.DataFrame, spec: SarimaxSpec, maxiter: int):
    model = SARIMAX(
        endog,
        exog=exog,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        trend=spec.trend,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=maxiter)


def choose_differencing(endog: pd.Series, d_grid: Sequence[int], alpha: float) -> tuple[int, float]:
    """ADF test on the quoted days only. d = 0 if a unit root is rejected at `alpha`, else 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pvalue = float(adfuller(endog.dropna().to_numpy(), autolag="AIC")[1])
    wanted = 0 if pvalue < alpha else 1
    d = min(d_grid, key=lambda v: (abs(v - wanted), v))
    return d, pvalue


def select_order(
    endog: pd.Series,
    exog: pd.DataFrame,
    d: int,
    grid: dict[str, Any],
    criterion: str = "aic",
    maxiter: int = 200,
) -> tuple[SarimaxSpec, list[dict[str, Any]]]:
    """Grid search over p, q, P, D, Q with d fixed, ranked by information criterion.

    AIC is only compared across candidates with the same differencing, so d is chosen first.
    """
    s = grid["seasonal_period"]
    candidates = []
    for p, q, P, D, Q in itertools.product(
        grid["order_grid"]["p"],
        grid["order_grid"]["q"],
        grid["seasonal_order_grid"]["P"],
        grid["seasonal_order_grid"]["D"],
        grid["seasonal_order_grid"]["Q"],
    ):
        seasonal = (P, D, Q, s) if (P, D, Q) != (0, 0, 0) else (0, 0, 0, 0)
        spec = SarimaxSpec(order=(p, d, q), seasonal_order=seasonal, trend="c" if d + D == 0 else None)
        record: dict[str, Any] = asdict(spec)
        try:
            res = _fit(endog, exog, spec, maxiter)
            record[criterion] = float(getattr(res, criterion))
            record["converged"] = bool(res.mle_retvals.get("converged", True))
        except (ValueError, np.linalg.LinAlgError) as exc:
            record[criterion] = float("nan")
            record["converged"] = False
            record["error"] = str(exc)
        candidates.append(record)

    ranked = [c for c in candidates if np.isfinite(c[criterion])]
    if not ranked:
        raise RuntimeError("No SARIMAX candidate could be fitted")
    converged = [c for c in ranked if c["converged"]]
    if not converged:
        logger.warning("No SARIMAX candidate converged; choosing the lowest %s anyway", criterion)
    best = min(converged or ranked, key=lambda c: c[criterion])
    spec = SarimaxSpec(
        order=tuple(best["order"]),
        seasonal_order=tuple(best["seasonal_order"]),
        trend=best["trend"],
    )
    return spec, candidates


def run_sarimax(cfg: dict[str, Any], data: ModelData) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one SARIMAX per crop on training rows and forecast every window in `predict_splits`."""
    roles = data.roles
    scfg = cfg["sarimax"]
    quantiles = cfg["forecast"]["quantiles"]
    predict_splits = cfg["forecast"]["predict_splits"]
    criterion = scfg["information_criterion"]
    maxiter = scfg["fit_maxiter"]
    group, target = roles["group"], roles["target"]

    frames, selection = [], {}
    for crop, g in data.frame.groupby(group, sort=True):
        g = g.set_index("date").sort_index().asfreq("D")
        price = g[target]
        exog = g[data.sarimax_exog]
        train = g["split"].eq("train")

        endog_train = to_endog(price[train], scfg["log_target"])
        d, adf_p = choose_differencing(endog_train, scfg["order_grid"]["d"], scfg["stationarity_alpha"])
        spec, candidates = select_order(endog_train, exog[train], d, scfg, criterion, maxiter)
        forecaster = SarimaxForecaster(spec, scfg["log_target"], maxiter).fit(price[train], exog[train])
        logger.info(
            "SARIMAX %s: ADF p=%.3f -> d=%d; selected order=%s seasonal=%s trend=%s %s=%.1f",
            crop, adf_p, d, spec.order, spec.seasonal_order, spec.trend, criterion,
            getattr(forecaster.result, criterion),
        )
        selection[crop] = {
            "adf_pvalue": adf_p,
            "d": d,
            **asdict(spec),
            criterion: float(getattr(forecaster.result, criterion)),
            "converged": bool(forecaster.result.mle_retvals.get("converged", True)),
            "params": {k: float(v) for k, v in forecaster.result.params.items()},
            "train_start": str(price[train].index.min().date()),
            "train_end": str(price[train].index.max().date()),
            "candidates": candidates,
        }

        windows = data.windows[data.windows[group].eq(crop) & data.windows["split"].isin(predict_splits)]
        for w in windows.itertuples(index=False):
            future = slice(w.decoder_start, w.decoder_end)
            values = forecaster.forecast(
                price.loc[: w.origin], exog.loc[: w.origin], exog.loc[future], quantiles
            )
            frames.append(
                forecast_rows(
                    MODEL_NAME, crop, w.split, w.origin, g.loc[future].reset_index(),
                    values, quantiles, target, roles["observed_flag"],
                )
            )
        logger.info("SARIMAX %s: forecast %d windows", crop, len(windows))

    return finalize_forecasts(frames, quantiles), selection
