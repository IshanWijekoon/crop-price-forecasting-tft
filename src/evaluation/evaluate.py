"""Score both models' forecast tables, test the difference, and plot.

Only rows with price_observed == 1 are scored. The point forecast is q50.

Inputs (paths.artifacts/forecasts/)
  {model}_forecasts.csv               protocol "train_only": parameters estimated once on the train split
  {model}_walkforward_forecasts.csv   protocol "walk_forward": re-estimated every refit_every_days

Outputs (paths.artifacts/evaluation/)
  metrics.csv              per protocol, split, model, crop (and "all"): point, quantile, and interval metrics
  metrics_by_horizon.csv   the same per horizon bucket
  diebold_mariano.csv      TFT vs SARIMAX per protocol, split, crop (and "all"), and loss
  plots/forecast_{crop}.png, plots/horizon_profile.png
  summary.md
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    absolute_errors,
    calculate_interval_width,
    calculate_mae,
    calculate_mape,
    calculate_picp,
    calculate_quantile_loss,
    calculate_rmse,
    pinball_losses,
)
from src.evaluation.statistical_tests import diebold_mariano_test
from src.models.forecast_output import quantile_columns
from src.utils.helpers import ensure_dir, load_config

logger = logging.getLogger(__name__)

MODELS = ("sarimax", "tft")
PROTOCOLS = {"train_only": "", "walk_forward": "_walkforward"}
ALL = "all"


def load_forecasts(forecast_dir: Path) -> pd.DataFrame:
    """Stack every protocol for which both models have a forecast table."""
    frames = []
    for protocol, suffix in PROTOCOLS.items():
        paths = {m: forecast_dir / f"{m}{suffix}_forecasts.csv" for m in MODELS}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            logger.warning("Skipping protocol %s: missing %s", protocol, missing)
            continue
        for path in paths.values():
            frames.append(pd.read_csv(path, parse_dates=["origin", "date"]).assign(protocol=protocol))
    if not frames:
        raise FileNotFoundError(f"No complete pair of forecast tables in {forecast_dir}")
    return pd.concat(frames, ignore_index=True)


def score(rows: pd.DataFrame, quantiles: Sequence[float], interval: Sequence[float]) -> dict[str, float]:
    qcols = quantile_columns(quantiles)
    lo, hi = quantile_columns(interval)
    y, median = rows["y_true"], rows["q50"]
    out: dict[str, float] = {
        "n_scored": len(rows),
        "n_origins": rows["origin"].nunique(),
        "mae": calculate_mae(y, median),
        "rmse": calculate_rmse(y, median),
        "mape_pct": calculate_mape(y, median),
    }
    for q, col in zip(quantiles, qcols):
        out[f"pinball_{col}"] = calculate_quantile_loss(y, rows[col], q)
    out["quantile_loss"] = float(np.mean([out[f"pinball_{c}"] for c in qcols]))
    out["picp"] = calculate_picp(y, rows[lo], rows[hi])
    out["picp_nominal"] = round(interval[1] - interval[0], 6)
    out["interval_width"] = calculate_interval_width(rows[lo], rows[hi])
    return out


def metric_table(
    scored: pd.DataFrame, keys: list[str], quantiles: Sequence[float], interval: Sequence[float]
) -> pd.DataFrame:
    """Metrics per `keys` + crop, plus a pooled crop == "all" row per `keys`."""
    rows = []
    for crop_keys in (keys + ["crop"], keys):
        for values, g in scored.groupby(crop_keys, sort=True):
            labels = dict(zip(crop_keys, values if isinstance(values, tuple) else (values,)))
            labels.setdefault("crop", ALL)
            rows.append({**labels, **score(g, quantiles, interval)})
    table = pd.DataFrame(rows)
    order = keys + ["crop"]
    table["_all"] = table["crop"].eq(ALL)
    return table.sort_values(order[:-1] + ["_all", "crop"]).drop(columns="_all").reset_index(drop=True)


def horizon_bucket(horizon_day: pd.Series, edges: Sequence[int]) -> pd.Series:
    bins = [0, *edges]
    labels = [f"{a + 1}-{b}" for a, b in zip(bins[:-1], bins[1:])]
    return pd.cut(horizon_day, bins=bins, labels=labels).astype(str)


def row_losses(rows: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    qcols = quantile_columns(quantiles)
    pinball = np.mean([pinball_losses(rows["y_true"], rows[c], q) for q, c in zip(quantiles, qcols)], axis=0)
    return pd.DataFrame(
        {"absolute_error": absolute_errors(rows["y_true"], rows["q50"]), "pinball": pinball},
        index=rows.index,
    )


def dm_tests(
    scored: pd.DataFrame, quantiles: Sequence[float], losses: Sequence[str], h: int, alpha: float
) -> pd.DataFrame:
    """TFT (A) vs SARIMAX (B) on the loss of each 60-day forecast.

    Each origin's loss is the mean over its quoted target days, so the series is indexed by
    forecast origin. Consecutive origins share up to h - 1 target days, hence lag h - 1.
    Pooled ("all") averages the crops' losses per origin.
    """
    keys = ["protocol", "split", "crop", "origin", "date"]
    scored = scored.join(row_losses(scored, quantiles))
    wide = scored.pivot_table(index=keys, columns="model", values=list(losses), aggfunc="first")
    if wide.isna().any().any():
        raise ValueError("Models were not scored on the same rows; DM test needs paired losses")

    per_origin = wide.groupby(level=["protocol", "split", "crop", "origin"]).mean()
    pooled = per_origin.groupby(level=["protocol", "split", "origin"]).mean()
    pooled = pooled.assign(crop=ALL).set_index("crop", append=True).reorder_levels(per_origin.index.names)
    series = pd.concat([per_origin, pooled]).sort_index()

    rows = []
    for (protocol, split, crop), g in series.groupby(level=["protocol", "split", "crop"], sort=True):
        for loss in losses:
            base = {"protocol": protocol, "split": split, "crop": crop, "loss": loss}
            try:
                res = diebold_mariano_test(g[(loss, "tft")], g[(loss, "sarimax")], h=h)
            except ValueError as exc:
                logger.warning("DM test skipped for %s: %s", base, exc)
                rows.append({**base, "note": str(exc)})
                continue
            significant = bool(res["p_value"] < alpha)
            rows.append(
                {
                    **base,
                    "n_origins": res["n"],
                    "h": res["h"],
                    "mean_loss_tft": res["mean_loss_a"],
                    "mean_loss_sarimax": res["mean_loss_b"],
                    "mean_diff_tft_minus_sarimax": res["mean_diff"],
                    "dm_stat": res["dm_stat"],
                    "hln_stat": res["hln_stat"],
                    "p_value": res["p_value"],
                    "variance_estimator": res["variance_estimator"],
                    "reject_h0": significant,
                    "lower_loss": "tft" if res["mean_diff"] < 0 else "sarimax",
                }
            )
    return pd.DataFrame(rows)


def plot_forecasts(
    scored: pd.DataFrame, forecasts: pd.DataFrame, protocol: str, split: str, horizon: int, out_dir: Path
) -> list[Path]:
    """Per crop: observed quotes and forecasts from origins `horizon` days apart, one panel per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc = forecasts[forecasts["protocol"].eq(protocol) & forecasts["split"].eq(split)]
    first = fc["origin"].min()
    shown = pd.date_range(first, fc["origin"].max(), freq=f"{horizon}D")
    paths = []
    for crop, g in fc.groupby("crop", sort=True):
        quotes = scored[
            scored["protocol"].eq(protocol) & scored["split"].eq(split) & scored["crop"].eq(crop)
        ].drop_duplicates("date")
        fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 3.4 * len(MODELS)), sharex=True, sharey=True)
        for ax, model in zip(axes, MODELS):
            m = g[g["model"].eq(model) & g["origin"].isin(shown)]
            for i, (origin, path) in enumerate(m.groupby("origin")):
                ax.fill_between(path["date"], path["q10"], path["q90"], color="tab:blue", alpha=0.2,
                                label="q10-q90" if i == 0 else None)
                ax.plot(path["date"], path["q50"], color="tab:blue", label="q50" if i == 0 else None)
                ax.axvline(origin, color="grey", ls="--", lw=0.8)
            ax.scatter(quotes["date"], quotes["y_true"], s=8, color="black", label="observed", zorder=3)
            ax.set_title(f"{model.upper()} - {crop} ({protocol}, {split})")
            ax.set_ylabel("LKR/kg")
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        path = out_dir / f"forecast_{crop}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_horizon_profile(scored: pd.DataFrame, protocol: str, split: str, nominal: float, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = scored[scored["protocol"].eq(protocol) & scored["split"].eq(split)]
    s = s.assign(abs_err=(s["y_true"] - s["q50"]).abs(), covered=s["y_true"].between(s["q10"], s["q90"]))
    prof = s.groupby(["model", "horizon_day"])[["abs_err", "covered"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for model, g in prof.groupby("model"):
        ax1.plot(g["horizon_day"], g["abs_err"], label=model.upper())
        ax2.plot(g["horizon_day"], g["covered"], label=model.upper())
    ax2.axhline(nominal, color="grey", ls="--", lw=1, label=f"nominal {nominal:.0%}")
    ax1.set(title="MAE by days ahead (all crops)", xlabel="days ahead", ylabel="LKR/kg")
    ax2.set(title="q10-q90 coverage by days ahead", xlabel="days ahead", ylabel="PICP", ylim=(0, 1))
    ax1.legend()
    ax2.legend()
    fig.suptitle(f"{protocol}, {split}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(
    metrics: pd.DataFrame, dm: pd.DataFrame, cfg: dict[str, Any], folds: dict[str, Any] | None, path: Path
) -> Path:
    ecfg = cfg["evaluation"]
    lines = [
        "# Evaluation summary",
        "",
        f"Horizon {cfg['forecast']['horizon']} days; scored on quoted days only; point forecast = q50; "
        f"PICP for the q{round(ecfg['interval'][0] * 100):02d}-q{round(ecfg['interval'][1] * 100):02d} "
        f"interval (nominal {ecfg['interval'][1] - ecfg['interval'][0]:.0%}).",
        "",
    ]
    if folds:
        lines += [
            f"Walk-forward over `{folds['split']}`: refit every {folds['refit_every_days']} days on an "
            f"expanding window ({len(folds['folds'])} refits, TFT {folds['tft']['epochs']} epochs per refit).",
            "",
            _markdown_table(pd.DataFrame(folds["folds"])),
            "",
        ]
    cols = ["protocol", "split", "model", "crop", "n_scored", "mae", "rmse", "mape_pct",
            "quantile_loss", "picp", "interval_width"]
    lines += ["## Metrics", "", _markdown_table(metrics[cols]), ""]
    if not dm.empty and "p_value" in dm:
        dcols = ["protocol", "split", "crop", "loss", "n_origins", "mean_loss_tft", "mean_loss_sarimax",
                 "dm_stat", "hln_stat", "p_value", "reject_h0", "lower_loss"]
        lines += [
            f"## Diebold-Mariano (TFT vs SARIMAX, two-sided, alpha = {ecfg['dm_alpha']})",
            "",
            "One loss per forecast origin (mean over its quoted target days), lag h-1 = "
            f"{int(dm['h'].dropna().iloc[0]) - 1} autocovariances. `dm_stat` is uncorrected; the p-value "
            "uses the HLN-corrected `hln_stat` against t(n-1). Negative statistic = TFT has lower loss.",
            "",
            _markdown_table(dm[[c for c in dcols if c in dm]]),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_evaluation(config_path: str | Path = "configs/config.yaml") -> dict[str, Path]:
    cfg = load_config(config_path)
    ecfg = cfg["evaluation"]
    quantiles, horizon = cfg["forecast"]["quantiles"], cfg["forecast"]["horizon"]
    interval = ecfg["interval"]
    if 0.5 not in quantiles or not set(interval) <= set(quantiles):
        raise ValueError("forecast.quantiles must include 0.5 and both evaluation.interval bounds")

    artifacts = Path(cfg["paths"]["artifacts"])
    out_dir = ensure_dir(artifacts / "evaluation")
    plot_dir = ensure_dir(out_dir / "plots")

    forecasts = load_forecasts(artifacts / "forecasts")
    scored = forecasts[forecasts["price_observed"].eq(1)].copy()
    if scored["y_true"].isna().any():
        raise ValueError("Rows flagged price_observed == 1 have no y_true")
    scored["horizon_bucket"] = horizon_bucket(scored["horizon_day"], ecfg["horizon_buckets"])

    metrics = metric_table(scored, ["protocol", "split", "model"], quantiles, interval)
    by_horizon = metric_table(scored, ["protocol", "split", "model", "horizon_bucket"], quantiles, interval)
    step = cfg["forecast"]["origin_step"]
    dm = dm_tests(scored, quantiles, ecfg["dm_losses"], h=math.ceil(horizon / step), alpha=ecfg["dm_alpha"])

    paths = {
        "metrics": out_dir / "metrics.csv",
        "metrics_by_horizon": out_dir / "metrics_by_horizon.csv",
        "diebold_mariano": out_dir / "diebold_mariano.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    by_horizon.to_csv(paths["metrics_by_horizon"], index=False)
    dm.to_csv(paths["diebold_mariano"], index=False)

    folds_path = artifacts / "forecasts" / "walk_forward_folds.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.exists() else None
    protocol = "walk_forward" if scored["protocol"].eq("walk_forward").any() else "train_only"
    split = folds["split"] if folds and protocol == "walk_forward" else "test"
    for i, p in enumerate(plot_forecasts(scored, forecasts, protocol, split, horizon, plot_dir)):
        paths[f"plot_forecast_{i}"] = p
    paths["plot_horizon"] = plot_horizon_profile(
        scored, protocol, split, interval[1] - interval[0], plot_dir / "horizon_profile.png"
    )
    paths["summary"] = write_summary(metrics, dm, cfg, folds, out_dir / "summary.md")

    for name, path in paths.items():
        logger.info("Wrote %s -> %s", name, path)
    pooled = metrics[metrics["crop"].eq(ALL)]
    for r in pooled.itertuples(index=False):
        logger.info(
            "%s %s %s: MAE=%.1f RMSE=%.1f MAPE=%.1f%% QL=%.2f PICP=%.2f",
            r.protocol, r.split, r.model, r.mae, r.rmse, r.mape_pct, r.quantile_loss, r.picp,
        )
    return paths
