"""Pipeline stages that fit each model and write its forecast table.

Outputs (paths.artifacts)
  forecasts/{model}_forecasts.csv   schema in src/models/forecast_output.py
  models/sarimax/selection.json     ADF test, order grid with AIC, selected order and parameters
  models/tft/tft-best.ckpt          best checkpoint by validation loss, plus logs/ with the loss curve
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.features.prepare_model_data import prepare_model_data
from src.utils.helpers import ensure_dir, load_config

logger = logging.getLogger(__name__)


def _write_forecasts(forecasts: pd.DataFrame, artifacts: Path, model: str) -> Path:
    path = ensure_dir(artifacts / "forecasts") / f"{model}_forecasts.csv"
    forecasts.to_csv(path, index=False, date_format="%Y-%m-%d")
    counts = forecasts.groupby(["split", "crop"])["origin"].nunique().to_dict()
    logger.info("Wrote %s forecasts -> %s (origins per split and crop: %s)", model, path, counts)
    return path


def run_sarimax_stage(config_path: str | Path = "configs/config.yaml") -> Path:
    from src.models.sarimax.model import MODEL_NAME, run_sarimax

    cfg = load_config(config_path)
    artifacts = Path(cfg["paths"]["artifacts"])
    forecasts, selection = run_sarimax(cfg, prepare_model_data(cfg))
    selection_path = ensure_dir(artifacts / "models" / MODEL_NAME) / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    logger.info("Wrote SARIMAX order selection -> %s", selection_path)
    return _write_forecasts(forecasts, artifacts, MODEL_NAME)


def run_tft_stage(config_path: str | Path = "configs/config.yaml") -> Path:
    # Imported lazily: torch and pytorch-forecasting take most of a minute to load.
    from src.models.tft.model import MODEL_NAME, run_tft

    cfg = load_config(config_path)
    artifacts = Path(cfg["paths"]["artifacts"])
    forecasts = run_tft(cfg, prepare_model_data(cfg), artifacts / "models" / MODEL_NAME)
    return _write_forecasts(forecasts, artifacts, MODEL_NAME)
