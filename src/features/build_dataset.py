"""Build the observed-data files under data/processed/.

Outputs
  price_observations.csv  every real price quote used or checked (CBSL daily, WFP monthly)
  covariates_daily.csv    one row per district per day: weather, dekadal rain, fuel, FX, calendar
  modeling_panel.csv      one row per crop per calendar day, Dambulla wholesale target
  build_report.json       coverage, anomaly decisions, and split boundaries

No price is generated or interpolated. Closed-market days keep a missing target.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.lag_features import generate_lag_features
from src.features.rolling_features import generate_observation_rolling_features, generate_rolling_features
from src.features.temporal_features import generate_temporal_features
from src.preprocessing.handle_missing_values import handle_missing_values
from src.preprocessing.load_sources import (
    WEATHER_COLUMNS,
    load_cbsl_daily_prices,
    load_dekadal_rainfall,
    load_fuel_daily,
    load_fx_daily,
    load_weather_daily,
    load_wfp_prices,
)
from src.preprocessing.merge_datasets import align_on_date, merge_datasets
from src.preprocessing.synchronize_time_series import synchronize_time_series
from src.utils.helpers import ensure_dir, load_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

TARGET = "price_lkr_per_kg"


def build_price_observations(cfg: dict[str, Any], raw: Path) -> pd.DataFrame:
    sources = cfg["dataset"]["sources"]
    cbsl = load_cbsl_daily_prices(raw / sources["cbsl_prices"])
    wfp = load_wfp_prices(raw / sources["wfp_prices"], cfg["dataset"]["wfp_commodities"])
    obs = pd.concat([cbsl, wfp], ignore_index=True)
    return obs.sort_values(["source", "crop", "market", "price_type", "date"]).reset_index(drop=True)


def flag_price_anomalies(
    obs: pd.DataFrame,
    market: str,
    corroborating_market: str,
    price_type: str,
    z_threshold: float,
) -> pd.DataFrame:
    """Flag isolated spike-and-revert quotes that the corroborating market does not show.

    A quote is flagged when the log change into it and out of it are both beyond
    `z_threshold` robust z-scores with opposite signs, and the corroborating
    market's same-day change is absent, opposite in sign, or under a quarter of the size.
    """
    obs = obs.copy()
    obs["anomaly_flag"] = 0
    cbsl = obs["source"].eq("CBSL") & obs["price_type"].eq(price_type)

    for crop in obs.loc[cbsl & obs["market"].eq(market), "crop"].unique():
        idx = obs.index[cbsl & obs["market"].eq(market) & obs["crop"].eq(crop)]
        series = obs.loc[idx].set_index("date")[TARGET].sort_index()
        log_p = np.log(series)
        change_in = log_p.diff()
        change_out = log_p.shift(-1) - log_p
        centred = change_in - change_in.median()
        scale = 1.4826 * centred.abs().median()
        z_in = centred / scale
        z_out = (change_out - change_in.median()) / scale
        spike = (z_in.abs() > z_threshold) & (z_out.abs() > z_threshold) & (np.sign(z_in) != np.sign(z_out))

        ref = obs[cbsl & obs["market"].eq(corroborating_market) & obs["crop"].eq(crop)]
        ref_change = np.log(ref.set_index("date")[TARGET].sort_index()).diff().reindex(series.index)
        corroborated = (np.sign(ref_change) == np.sign(change_in)) & (ref_change.abs() >= 0.25 * change_in.abs())

        flagged_dates = series.index[(spike & ~corroborated).to_numpy()]
        obs.loc[idx[obs.loc[idx, "date"].isin(flagged_dates).to_numpy()], "anomaly_flag"] = 1
        logger.info(
            "%s %s: %d large jumps, %d flagged as uncorroborated spikes",
            market,
            crop,
            int((z_in.abs() > z_threshold).sum()),
            len(flagged_dates),
        )
    return obs


def build_covariates_daily(cfg: dict[str, Any], raw: Path, market_open: pd.Series) -> pd.DataFrame:
    ds = cfg["dataset"]
    start, end = ds["covariate_start"], ds["covariate_end"]
    sources = ds["sources"]

    weather = load_weather_daily(raw / sources["weather_daily"])
    weather = weather[weather["district_key"].isin(ds["districts"])]
    weather = synchronize_time_series(weather, group_cols=["district_key"], start=start, end=end)
    weather = handle_missing_values(
        weather, WEATHER_COLUMNS, strategy="interpolate_ffill", group_cols=["district_key"], flag_suffix="_imputed"
    )
    imputed_cols = [f"{c}_imputed" for c in WEATHER_COLUMNS]
    weather["weather_imputed"] = weather[imputed_cols].max(axis=1).astype("int8")
    weather = weather.drop(columns=imputed_cols)
    weather = generate_rolling_features(
        weather,
        columns=["rainfall_mm"],
        windows=ds["rainfall_rolling_windows"],
        stats=["sum"],
        group_cols=["district_key"],
        min_periods={w: w for w in ds["rainfall_rolling_windows"]},
        name="rainfall",
    )

    dekad = load_dekadal_rainfall(raw / sources["dekadal_rainfall"], {v: k for k, v in ds["districts"].items()})
    weather = weather.sort_values("date")
    weather = pd.merge_asof(
        weather,
        dekad.sort_values("available_from")[
            ["district_key", "available_from", "rain_dekad_mm", "rain_dekad_anomaly_pct"]
        ],
        left_on="date",
        right_on="available_from",
        by="district_key",
        direction="backward",
    ).drop(columns="available_from")

    fuel = load_fuel_daily(
        raw / sources["fuel_revisions"], start, end, check_path=raw / sources["fuel_daily_check"]
    )
    fx = load_fx_daily(raw / sources["usd_lkr_daily"], start, end)
    national = merge_datasets([fuel, fx], on=["date"], how="inner")

    calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    weekday_rule = (calendar["date"].dt.dayofweek < 5).astype("int8")
    observed_open = calendar["date"].map(market_open)
    calendar["market_open"] = observed_open.fillna(weekday_rule).astype("int8")
    calendar["market_open_source"] = np.where(observed_open.notna(), "cbsl_quote", "weekday_rule")
    calendar = generate_temporal_features(calendar)
    national = merge_datasets([national, calendar], on=["date"], how="inner")

    cov = align_on_date(weather, national)
    return cov.sort_values(["district_key", "date"]).reset_index(drop=True)


def build_modeling_panel(
    cfg: dict[str, Any],
    obs: pd.DataFrame,
    cov: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ds = cfg["dataset"]
    tgt = ds["target"]

    target = obs[
        obs["source"].eq(tgt["source"])
        & obs["market"].eq(tgt["market"])
        & obs["price_type"].eq(tgt["price_type"])
        & obs["crop"].isin(tgt["crops"])
    ].copy()
    target.loc[target["anomaly_flag"].eq(1), TARGET] = np.nan

    start = target["date"].min()
    end = min(target["date"].max(), pd.Timestamp(ds["covariate_end"]))
    panel = synchronize_time_series(
        target[["date", "crop", TARGET]], group_cols=["crop"], start=start, end=end
    )
    panel["market"] = tgt["market"]
    panel["price_type"] = tgt["price_type"]
    panel["time_idx"] = (panel["date"] - start).dt.days
    panel["price_observed"] = panel[TARGET].notna().astype("int8")
    panel["price_ffill"] = panel.groupby("crop")[TARGET].ffill()
    panel["price_is_carried_forward"] = (panel["price_observed"].eq(0) & panel["price_ffill"].notna()).astype("int8")

    panel = generate_lag_features(panel, ["price_ffill"], ds["price_lags"], group_cols=["crop"])
    panel = panel.rename(columns={f"price_ffill_lag_{k}": f"price_lag_{k}" for k in ds["price_lags"]})
    panel = generate_observation_rolling_features(
        panel,
        column=TARGET,
        windows=ds["price_rolling_quotes"],
        stats=["mean", "std"],
        group_cols=["crop"],
        name="price",
    )

    district_cols = [
        *WEATHER_COLUMNS,
        "rainfall_roll_sum_7",
        "rainfall_roll_sum_30",
        "rain_dekad_mm",
        "rain_dekad_anomaly_pct",
        "weather_imputed",
    ]
    wide = cov.pivot(index="date", columns="district_key", values=district_cols)
    wide.columns = [f"{district}_{col}" for col, district in wide.columns]
    national_cols = [
        "diesel_lkr_per_litre",
        "petrol92_lkr_per_litre",
        "fuel_revision_day",
        "usd_lkr",
        "usd_lkr_carried_forward",
        "market_open",
        "year",
        "month",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "month_sin",
        "month_cos",
        "season",
    ]
    national = cov.drop_duplicates("date").set_index("date")[national_cols]
    exog = national.join(wide).reset_index()
    panel = align_on_date(panel, exog)

    lag_cols = [f"price_lag_{k}" for k in ds["price_lags"]]
    roll_cols = [f"price_roll_{s}_{w}q" for w in ds["price_rolling_quotes"] for s in ("mean", "std")]
    feature_ready = panel[lag_cols + roll_cols].notna().all(axis=1)
    ready_by_date = feature_ready.groupby(panel["date"]).all()
    model_start = ready_by_date[ready_by_date].index.min()

    model_dates = pd.Series(sorted(panel.loc[panel["date"] >= model_start, "date"].unique()))
    n = len(model_dates)
    n_train = int(np.floor(n * cfg["splits"]["train"]))
    n_val = int(np.floor(n * cfg["splits"]["validation"]))
    bounds = {
        "train": (model_dates.iloc[0], model_dates.iloc[n_train - 1]),
        "validation": (model_dates.iloc[n_train], model_dates.iloc[n_train + n_val - 1]),
        "test": (model_dates.iloc[n_train + n_val], model_dates.iloc[-1]),
    }
    panel["split"] = "warmup"
    for name, (lo, hi) in bounds.items():
        panel.loc[panel["date"].between(lo, hi), "split"] = name

    id_cols = ["date", "time_idx", "crop", "market", "price_type"]
    target_cols = [TARGET, "price_observed", "price_ffill", "price_is_carried_forward"]
    ordered = id_cols + target_cols + lag_cols + roll_cols
    panel = panel[ordered + [c for c in panel.columns if c not in ordered and c != "split"] + ["split"]]
    panel = panel.sort_values(["crop", "date"]).reset_index(drop=True)

    split_info = {
        name: {
            "start": str(lo.date()),
            "end": str(hi.date()),
            "days": int(panel.loc[panel["split"].eq(name), "date"].nunique()),
            "observed_target_rows": int(panel.loc[panel["split"].eq(name), "price_observed"].sum()),
        }
        for name, (lo, hi) in bounds.items()
    }
    split_info["warmup"] = {
        "start": str(start.date()),
        "end": str((model_start - pd.Timedelta(days=1)).date()),
        "days": int(panel.loc[panel["split"].eq("warmup"), "date"].nunique()),
    }
    return panel, split_info


def build_report(obs: pd.DataFrame, cov: pd.DataFrame, panel: pd.DataFrame, split_info: dict) -> dict:
    coverage = (
        obs.groupby(["source", "market", "price_type", "crop"])["date"]
        .agg(first="min", last="max", quotes="size")
        .reset_index()
    )
    coverage[["first", "last"]] = coverage[["first", "last"]].astype(str)
    target_rows = panel.groupby("crop").agg(
        rows=("date", "size"),
        observed=("price_observed", "sum"),
        mean_price=("price_lkr_per_kg", "mean"),
        min_price=("price_lkr_per_kg", "min"),
        max_price=("price_lkr_per_kg", "max"),
    )
    flagged = obs[obs["anomaly_flag"].eq(1)][["date", "market", "crop", "price_type", TARGET]]
    return {
        "target": "CBSL Dambulla wholesale, LKR/kg, daily",
        "panel_period": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "panel_rows": int(len(panel)),
        "target_by_crop": target_rows.round(2).reset_index().to_dict(orient="records"),
        "splits": split_info,
        "anomalies_removed_from_target": flagged.astype(str).to_dict(orient="records"),
        "covariates": {
            "rows": int(len(cov)),
            "period": [str(cov["date"].min().date()), str(cov["date"].max().date())],
            "weather_imputed_days": int(cov["weather_imputed"].sum()),
            "usd_lkr_carried_forward_days": int(cov.drop_duplicates("date")["usd_lkr_carried_forward"].sum()),
            "empty_cells": int(cov.drop(columns=["rain_dekad_mm", "rain_dekad_anomaly_pct"]).isna().sum().sum()),
        },
        "panel_empty_cells_outside_target_and_warmup": int(
            panel.loc[panel["split"].ne("warmup")]
            .drop(columns=["price_lkr_per_kg"])
            .isna()
            .sum()
            .sum()
        ),
        "price_observation_coverage": coverage.to_dict(orient="records"),
    }


def build_dataset(config_path: str | Path = "configs/config.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw"])
    out_dir = ensure_dir(cfg["paths"]["processed"])
    tgt = cfg["dataset"]["target"]

    obs = build_price_observations(cfg, raw)
    obs = flag_price_anomalies(
        obs,
        market=tgt["market"],
        corroborating_market=tgt["corroborating_market"],
        price_type=tgt["price_type"],
        z_threshold=cfg["dataset"]["anomaly_mad_z"],
    )

    target_quotes = obs[
        obs["source"].eq(tgt["source"]) & obs["market"].eq(tgt["market"]) & obs["price_type"].eq(tgt["price_type"])
    ]
    open_days = pd.date_range(target_quotes["date"].min(), target_quotes["date"].max(), freq="D")
    market_open = pd.Series(open_days.isin(target_quotes["date"]).astype("int8"), index=open_days)

    cov = build_covariates_daily(cfg, raw, market_open)
    panel, split_info = build_modeling_panel(cfg, obs, cov)
    report = build_report(obs, cov, panel, split_info)

    paths = {
        "price_observations": out_dir / "price_observations.csv",
        "covariates_daily": out_dir / "covariates_daily.csv",
        "modeling_panel": out_dir / "modeling_panel.csv",
        "build_report": out_dir / "build_report.json",
    }
    obs.to_csv(paths["price_observations"], index=False, date_format="%Y-%m-%d")
    cov.to_csv(paths["covariates_daily"], index=False, date_format="%Y-%m-%d")
    panel.to_csv(paths["modeling_panel"], index=False, date_format="%Y-%m-%d")
    paths["build_report"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for name, path in paths.items():
        logger.info("Wrote %s -> %s", name, path)
    logger.info("Splits: %s", {k: (v["start"], v["end"], v["days"]) for k, v in split_info.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the observed-data modeling panel.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    setup_logger("src")
    setup_logger(__name__)
    build_dataset(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
