"""Temporal Fusion Transformer (PyTorch Forecasting), one global model across crops.

The network target is the carried-forward price (`features.target_filled`), because the
dataset cannot hold missing targets. `features.observed_flag` is the loss weight, so days
without a quote contribute nothing to training or early stopping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.features.prepare_model_data import ModelData
from src.models.forecast_output import finalize_forecasts, forecast_rows
from src.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

MODEL_NAME = "tft"
CHECKPOINT_NAME = "tft-best"


def model_frame(data: ModelData) -> pd.DataFrame:
    roles = data.roles
    df = data.frame.copy()
    df[roles["time_idx"]] = df[roles["time_idx"]].astype(int)
    for col in roles["static_categoricals"] + roles["known_categoricals"]:
        df[col] = df[col].astype(str)
    flags = roles["known_flags"] + roles["observed_flags"] + [roles["observed_flag"]]
    df[flags] = df[flags].astype(float)
    df[roles["target_filled"]] = df[roles["target_filled"]].astype(float)
    return df


def training_dataset(cfg: dict[str, Any], roles: dict[str, Any], rows: pd.DataFrame) -> TimeSeriesDataSet:
    """Every full encoder/decoder window inside `rows` (a `model_frame` slice)."""
    tcfg = cfg["tft"]
    encoder, horizon = cfg["forecast"]["window_size"], cfg["forecast"]["horizon"]
    time_idx, group = roles["time_idx"], roles["group"]
    return TimeSeriesDataSet(
        rows,
        time_idx=time_idx,
        target=roles["target_filled"],
        group_ids=[group],
        weight=roles["observed_flag"],
        max_encoder_length=encoder,
        min_encoder_length=encoder,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        static_categoricals=roles["static_categoricals"],
        time_varying_known_categoricals=roles["known_categoricals"],
        time_varying_known_reals=roles["known_reals"] + roles["known_flags"],
        time_varying_unknown_reals=[roles["target_filled"]] + roles["observed_reals"] + roles["observed_flags"],
        target_normalizer=GroupNormalizer(groups=[group], transformation=tcfg["target_transformation"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=False,
    )


def build_datasets(cfg: dict[str, Any], data: ModelData) -> dict[str, TimeSeriesDataSet]:
    """Training windows from the train split; validation and test windows reuse its encoders.

    A split's windows have their whole decoder inside that split, matching
    `forecast_windows.csv`. Encoders may reach back into earlier splits.
    """
    roles = data.roles
    time_idx = roles["time_idx"]
    df = model_frame(data)
    training = training_dataset(cfg, roles, df[df["split"].eq("train")])

    datasets = {"train": training}
    order = ["train", "validation", "test"]
    for split in order[1:]:
        upto = order[: order.index(split) + 1]
        rows = df[df["split"].isin(upto)]
        first_idx = int(df.loc[df["split"].eq(split), time_idx].min())
        datasets[split] = TimeSeriesDataSet.from_dataset(
            training, rows, min_prediction_idx=first_idx, stop_randomization=True
        )
    for split, ds in datasets.items():
        logger.info("TFT %s windows: %d", split, len(ds))
    return datasets


def _new_model(cfg: dict[str, Any], training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    tcfg = cfg["tft"]
    seed_everything(cfg["random_seed"], workers=True)
    return TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=tcfg["hidden_size"],
        attention_head_size=tcfg["attention_head_size"],
        hidden_continuous_size=tcfg["hidden_continuous_size"],
        lstm_layers=tcfg["lstm_layers"],
        dropout=tcfg["dropout"],
        learning_rate=tcfg["learning_rate"],
        loss=QuantileLoss(quantiles=cfg["forecast"]["quantiles"]),
        reduce_on_plateau_patience=tcfg["reduce_on_plateau_patience"],
        log_interval=-1,
    )


def train_tft(
    cfg: dict[str, Any], datasets: dict[str, TimeSeriesDataSet], out_dir: Path
) -> TemporalFusionTransformer:
    tcfg = cfg["tft"]
    model = _new_model(cfg, datasets["train"])
    logger.info("TFT parameters: %d", sum(p.numel() for p in model.parameters()))

    checkpoint = ModelCheckpoint(
        dirpath=out_dir,
        filename=CHECKPOINT_NAME,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
    )
    trainer = Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator="cpu",
        gradient_clip_val=tcfg["gradient_clip_val"],
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=tcfg["early_stopping_patience"]),
            checkpoint,
        ],
        logger=CSVLogger(save_dir=out_dir, name="logs"),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    batch = tcfg["batch_size"]
    trainer.fit(
        model,
        train_dataloaders=datasets["train"].to_dataloader(train=True, batch_size=batch, num_workers=0),
        val_dataloaders=datasets["validation"].to_dataloader(train=False, batch_size=batch, num_workers=0),
    )
    logger.info(
        "TFT stopped after %d epochs; best val_loss=%.4f at %s",
        trainer.current_epoch, float(checkpoint.best_model_score), checkpoint.best_model_path,
    )
    return TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)


def checkpoint_epochs(path: str | Path) -> int:
    """Number of epochs the saved model had trained for when its checkpoint was written."""
    import torch

    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) + 1


def refit_tft(
    cfg: dict[str, Any], training: TimeSeriesDataSet, epochs: int, log_dir: Path | None = None
) -> TemporalFusionTransformer:
    """Fresh model trained for a fixed number of epochs, with no validation data held back."""
    model = _new_model(cfg, training)
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        gradient_clip_val=cfg["tft"]["gradient_clip_val"],
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=CSVLogger(save_dir=log_dir, name="logs") if log_dir is not None else False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        model,
        train_dataloaders=training.to_dataloader(
            train=True, batch_size=cfg["tft"]["batch_size"], num_workers=0
        ),
    )
    return model


def predict_dataset(
    cfg: dict[str, Any],
    data: ModelData,
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    split: str,
) -> list[pd.DataFrame]:
    """Forecast rows (unfinalized) for every window in `dataset`, labelled with `split`."""
    roles = data.roles
    quantiles = cfg["forecast"]["quantiles"]
    horizon = cfg["forecast"]["horizon"]
    time_idx, group = roles["time_idx"], roles["group"]
    by_group = {crop: g.set_index(time_idx).sort_index() for crop, g in data.frame.groupby(group)}

    pred = model.predict(
        dataset.to_dataloader(train=False, batch_size=cfg["tft"]["batch_size"], num_workers=0),
        mode="quantiles",
        return_index=True,
        trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_progress_bar": False},
    )
    values = pred.output.detach().cpu().numpy()
    frames = []
    for i, row in enumerate(pred.index.itertuples(index=False)):
        g = by_group[getattr(row, group)]
        first = int(getattr(row, time_idx))
        decoder = g.loc[first : first + horizon - 1].reset_index()
        origin = g.loc[first - 1, "date"]
        frames.append(
            forecast_rows(
                MODEL_NAME, getattr(row, group), split, origin, decoder,
                values[i], quantiles, roles["target"], roles["observed_flag"],
            )
        )
    return frames


def predict_tft(
    cfg: dict[str, Any],
    data: ModelData,
    model: TemporalFusionTransformer,
    datasets: dict[str, TimeSeriesDataSet],
) -> pd.DataFrame:
    frames = []
    for split in cfg["forecast"]["predict_splits"]:
        split_frames = predict_dataset(cfg, data, model, datasets[split], split)
        logger.info("TFT forecast %d %s windows", len(split_frames), split)
        frames.extend(split_frames)
    return finalize_forecasts(frames, cfg["forecast"]["quantiles"])


def run_tft(cfg: dict[str, Any], data: ModelData, out_dir: Path) -> pd.DataFrame:
    datasets = build_datasets(cfg, data)
    model = train_tft(cfg, datasets, ensure_dir(out_dir))
    return predict_tft(cfg, data, model, datasets)
