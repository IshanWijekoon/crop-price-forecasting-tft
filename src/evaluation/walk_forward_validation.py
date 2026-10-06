"""Expanding-window walk-forward forecasts for both models.

Folds cover the forecast origins of `evaluation.walk_forward.split`. At each fold's refit
date both models are re-estimated on every row up to and including that date, then forecast
60 days from each daily origin until the next refit, using all data known at that origin.
Hyperparameters stay as tuned on the training and validation splits: SARIMAX orders from
`selection.json`, and the TFT epoch count of the validation-selected checkpoint.

Outputs (paths.artifacts)
  forecasts/{model}_walkforward_forecasts.csv   schema in src/models/forecast_output.py
  forecasts/walk_forward_folds.json             fold dates, SARIMAX parameters and TFT settings per fold
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.features.prepare_model_data import ModelData, prepare_model_data
from src.models.forecast_output import finalize_forecasts, forecast_rows
from src.models.train import write_forecasts
from src.utils.helpers import ensure_dir, load_config

logger = logging.getLogger(__name__)

SUFFIX = "_walkforward"


@dataclass(frozen=True)
class Fold:
    fold: int
    refit_date: pd.Timestamp
    origins: tuple[pd.Timestamp, ...]

    def record(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "refit_date": str(self.refit_date.date()),
            "first_origin": str(self.origins[0].date()),
            "last_origin": str(self.origins[-1].date()),
            "origins": len(self.origins),
        }


def expanding_window_folds(origins: Sequence[pd.Timestamp], refit_every_days: int) -> list[Fold]:
    """Group sorted origins into folds. A fold's refit date is its first origin; the next fold
    starts at the first origin at least `refit_every_days` later."""
    if refit_every_days < 1:
        raise ValueError("refit_every_days must be >= 1")
    ordered = sorted(pd.DatetimeIndex(origins).unique())
    if not ordered:
        raise ValueError("No forecast origins to fold")
    groups: list[list[pd.Timestamp]] = []
    for origin in ordered:
        if not groups or (origin - groups[-1][0]).days >= refit_every_days:
            groups.append([])
        groups[-1].append(origin)
    return [Fold(i, g[0], tuple(g)) for i, g in enumerate(groups)]


def walk_forward_sarimax(
    cfg: dict[str, Any], data: ModelData, folds: list[Fold], selection: dict[str, Any], split: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from src.models.sarimax.model import MODEL_NAME, SarimaxForecaster, SarimaxSpec

    roles, scfg = data.roles, cfg["sarimax"]
    quantiles, horizon = cfg["forecast"]["quantiles"], cfg["forecast"]["horizon"]
    target, criterion = roles["target"], scfg["information_criterion"]

    frames, records = [], []
    for crop, g in data.frame.groupby(roles["group"], sort=True):
        if crop not in selection:
            raise KeyError(f"No SARIMAX order for {crop!r} in selection.json; rerun --stages sarimax")
        sel = selection[crop]
        spec = SarimaxSpec(tuple(sel["order"]), tuple(sel["seasonal_order"]), sel["trend"])
        g = g.set_index("date").sort_index().asfreq("D")
        price, exog = g[target], g[data.sarimax_exog]

        for fold in folds:
            history = slice(None, fold.refit_date)
            forecaster = SarimaxForecaster(spec, scfg["log_target"], scfg["fit_maxiter"])
            forecaster.fit(price.loc[history], exog.loc[history])
            records.append(
                {
                    "crop": crop,
                    "fold": fold.fold,
                    "refit_date": str(fold.refit_date.date()),
                    "quotes_used": int(price.loc[history].notna().sum()),
                    "order": list(spec.order),
                    "seasonal_order": list(spec.seasonal_order),
                    "trend": spec.trend,
                    criterion: float(getattr(forecaster.result, criterion)),
                    "converged": bool(forecaster.result.mle_retvals.get("converged", True)),
                    "params": {k: float(v) for k, v in forecaster.result.params.items()},
                }
            )
            for origin in fold.origins:
                future = slice(origin + pd.Timedelta(days=1), origin + pd.Timedelta(days=horizon))
                values = forecaster.forecast(
                    price.loc[:origin], exog.loc[:origin], exog.loc[future], quantiles
                )
                frames.append(
                    forecast_rows(
                        MODEL_NAME, crop, split, origin, g.loc[future].reset_index(),
                        values, quantiles, target, roles["observed_flag"],
                    )
                )
        logger.info("SARIMAX walk-forward %s: %d refits", crop, len(folds))
    return finalize_forecasts(frames, quantiles), records


def walk_forward_tft(
    cfg: dict[str, Any], data: ModelData, folds: list[Fold], epochs: int, split: str, log_dir: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from pytorch_forecasting import TimeSeriesDataSet

    from src.models.tft.model import model_frame, predict_dataset, refit_tft, training_dataset

    roles = data.roles
    time_idx, horizon = roles["time_idx"], cfg["forecast"]["horizon"]
    df = model_frame(data)
    idx_of = df.drop_duplicates("date").set_index("date")[time_idx]

    frames, records = [], []
    for fold in folds:
        cutoff = int(idx_of[fold.refit_date])
        first, last = int(idx_of[fold.origins[0]]), int(idx_of[fold.origins[-1]])
        training = training_dataset(cfg, roles, df[df[time_idx] <= cutoff])
        model = refit_tft(cfg, training, epochs, log_dir / f"fold_{fold.fold}")
        predict = TimeSeriesDataSet.from_dataset(
            training, df[df[time_idx] <= last + horizon], min_prediction_idx=first + 1, stop_randomization=True
        )
        fold_frames = predict_dataset(cfg, data, model, predict, split)
        frames.extend(fold_frames)
        records.append(
            {
                "fold": fold.fold,
                "refit_date": str(fold.refit_date.date()),
                "train_windows": len(training),
                "epochs": epochs,
                "forecast_windows": len(fold_frames),
            }
        )
        logger.info(
            "TFT walk-forward fold %d: refit on %d windows up to %s, forecast %d windows",
            fold.fold, len(training), fold.refit_date.date(), len(fold_frames),
        )
    return finalize_forecasts(frames, cfg["forecast"]["quantiles"]), records


def _check_origins(forecasts: pd.DataFrame, folds: list[Fold], model: str) -> None:
    expected = {o for f in folds for o in f.origins}
    for crop, g in forecasts.groupby("crop"):
        got = set(pd.DatetimeIndex(g["origin"].unique()))
        if got != expected:
            raise RuntimeError(
                f"{model} walk-forward {crop}: {len(got)} origins, expected {len(expected)}"
            )


def run_walk_forward(config_path: str | Path = "configs/config.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    wcfg = cfg["evaluation"]["walk_forward"]
    artifacts = Path(cfg["paths"]["artifacts"])
    split = wcfg["split"]
    data = prepare_model_data(cfg)

    windows = data.windows[data.windows["split"].eq(split)]
    per_crop = windows.groupby(data.roles["group"])["origin"].apply(frozenset)
    if per_crop.nunique() != 1:
        raise ValueError(f"Crops do not share the same {split} forecast origins")
    folds = expanding_window_folds(windows["origin"], wcfg["refit_every_days"])
    logger.info("Walk-forward over %s: %s", split, [f.record() for f in folds])

    selection_path = artifacts / "models" / "sarimax" / "selection.json"
    checkpoint_path = artifacts / "models" / "tft" / "tft-best.ckpt"
    if not selection_path.exists():
        raise FileNotFoundError(f"{selection_path} not found; run --stages sarimax first")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    sarimax_fc, sarimax_records = walk_forward_sarimax(cfg, data, folds, selection, split)
    _check_origins(sarimax_fc, folds, "sarimax")

    from src.models.tft.model import checkpoint_epochs

    if wcfg.get("tft_epochs") is not None:
        epochs, epochs_source = int(wcfg["tft_epochs"]), "config"
    elif checkpoint_path.exists():
        epochs, epochs_source = checkpoint_epochs(checkpoint_path), str(checkpoint_path)
    else:
        raise FileNotFoundError(f"{checkpoint_path} not found; run --stages tft or set tft_epochs")
    logger.info("TFT refits train for %d epochs (from %s)", epochs, epochs_source)
    tft_fc, tft_records = walk_forward_tft(
        cfg, data, folds, epochs, split, artifacts / "models" / "tft" / "walk_forward"
    )
    _check_origins(tft_fc, folds, "tft")

    paths = {
        "sarimax": write_forecasts(sarimax_fc, artifacts, "sarimax", SUFFIX),
        "tft": write_forecasts(tft_fc, artifacts, "tft", SUFFIX),
        "folds": ensure_dir(artifacts / "forecasts") / "walk_forward_folds.json",
    }
    record = {
        "split": split,
        "refit_every_days": wcfg["refit_every_days"],
        "horizon_days": cfg["forecast"]["horizon"],
        "folds": [f.record() for f in folds],
        "sarimax": sarimax_records,
        "tft": {"epochs": epochs, "epochs_source": epochs_source, "folds": tft_records},
    }
    paths["folds"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("Wrote walk-forward folds -> %s", paths["folds"])
    return paths
