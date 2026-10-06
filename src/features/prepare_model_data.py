"""Turn modeling_panel.csv into the scaled frame and forecast windows both models read.

Outputs (data/processed/)
  model_frame.csv        panel without warm-up rows, predictors Min-Max scaled on train rows
  forecast_windows.csv   one row per crop and forecast origin, with the split of its decoder
  minmax_scaler.json     per-column training min/max, plus the training period it was fitted on
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.lag_features import generate_lag_features
from src.features.normalize_features import normalize_features, scaler_to_dict
from src.features.windows import make_forecast_windows
from src.utils.helpers import ensure_dir, load_config

logger = logging.getLogger(__name__)

ROLE_KEYS = (
    "static_categoricals",
    "known_categoricals",
    "known_reals",
    "known_flags",
    "observed_reals",
    "observed_flags",
)


@dataclass
class ModelData:
    frame: pd.DataFrame
    windows: pd.DataFrame
    scaler: Any
    scaled_columns: list[str]
    sarimax_exog: list[str]
    roles: dict[str, Any]


def load_panel(path: str | Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date"])
    return panel.sort_values(["crop", "date"]).reset_index(drop=True)


def prepare_model_data(cfg: dict[str, Any], panel: pd.DataFrame | None = None) -> ModelData:
    roles = cfg["features"]
    group = roles["group"]
    if panel is None:
        panel = load_panel(Path(cfg["paths"]["processed"]) / "modeling_panel.csv")

    required = [roles["target"], roles["target_filled"], roles["observed_flag"], group, roles["time_idx"]]
    required += [c for key in ROLE_KEYS for c in roles[key]]
    required += cfg["sarimax"]["exog"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise KeyError(f"modeling_panel is missing configured columns: {missing}")

    # Shifted before the warm-up is dropped, so the first training rows can reach back into it.
    exog_lag = cfg["sarimax"]["exog_lag_days"]
    if exog_lag < cfg["forecast"]["horizon"]:
        raise ValueError("sarimax.exog_lag_days must be >= forecast.horizon to keep exog known at the origin")
    panel = generate_lag_features(panel, cfg["sarimax"]["exog"], [exog_lag], group_cols=[group])
    sarimax_exog = [f"{c}_lag_{exog_lag}" for c in cfg["sarimax"]["exog"]]

    frame = panel[panel["split"].ne("warmup")].reset_index(drop=True)

    predictors = [c for key in ROLE_KEYS for c in roles[key]] + sarimax_exog + [roles["target_filled"]]
    empty = frame[predictors].isna().sum()
    empty = empty[empty > 0]
    if not empty.empty:
        raise ValueError(f"Predictors have empty cells after warm-up: {empty.to_dict()}")

    scaled_columns = list(dict.fromkeys(roles["known_reals"] + roles["observed_reals"] + sarimax_exog))
    train_mask = frame["split"].eq("train")
    frame, scaler = normalize_features(
        frame, scaled_columns, method=roles["scaling"], fit_mask=train_mask
    )

    windows = make_forecast_windows(
        frame,
        encoder_length=cfg["forecast"]["window_size"],
        horizon=cfg["forecast"]["horizon"],
        step=cfg["forecast"]["origin_step"],
        group_column=group,
        observed_column=roles["observed_flag"],
    )
    return ModelData(
        frame=frame,
        windows=windows,
        scaler=scaler,
        scaled_columns=scaled_columns,
        sarimax_exog=sarimax_exog,
        roles=roles,
    )


def run_prepare(config_path: str | Path = "configs/config.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    out_dir = ensure_dir(cfg["paths"]["processed"])
    data = prepare_model_data(cfg)

    train = data.frame[data.frame["split"].eq("train")]
    scaler_record = {
        "method": data.roles["scaling"],
        "fitted_on": {
            "split": "train",
            "start": str(train["date"].min().date()),
            "end": str(train["date"].max().date()),
            "rows": int(len(train)),
        },
        "columns": scaler_to_dict(data.scaler, data.scaled_columns),
    }

    paths = {
        "model_frame": out_dir / "model_frame.csv",
        "forecast_windows": out_dir / "forecast_windows.csv",
        "minmax_scaler": out_dir / "minmax_scaler.json",
    }
    data.frame.to_csv(paths["model_frame"], index=False, date_format="%Y-%m-%d")
    data.windows.to_csv(paths["forecast_windows"], index=False, date_format="%Y-%m-%d")
    paths["minmax_scaler"].write_text(json.dumps(scaler_record, indent=2), encoding="utf-8")

    for name, path in paths.items():
        logger.info("Wrote %s -> %s", name, path)
    logger.info(
        "Windows per split and crop: %s",
        data.windows.groupby(["split", data.roles["group"]]).size().to_dict(),
    )
    return paths
