# Evaluation

Walk-forward forecasts, metrics, and the Diebold–Mariano test comparing TFT with SARIMAX on the same rows.

```bash
python -m src.main --stages walkforward evaluate   # needs the sarimax and tft stages to have run
python -m src.main --stages evaluate               # re-score existing forecast tables only
```

| File | Role |
|------|------|
| `walk_forward_validation.py` | `walkforward` stage: expanding-window refits of both models |
| `evaluate.py` | `evaluate` stage: metric tables, DM tests, plots, `summary.md` |
| `metrics.py` | MAE, RMSE, MAPE, pinball loss, PICP, interval width |
| `statistical_tests.py` | Diebold–Mariano test with the Harvey–Leybourne–Newbold correction |

## Protocols

Both protocols score every daily forecast origin whose whole 60-day decoder falls in the split (77 test origins per crop, 76 validation).

- **`walk_forward`** (methodology section 2.4.2). Origins in `evaluation.walk_forward.split` (test) are grouped into folds of `refit_every_days` (30). At the first origin of each fold, both models are re-estimated on every row up to that date, then forecast each origin in the fold with all data known at that origin. The window expands; nothing is dropped from the start.
  - SARIMAX: the order chosen on the training split (`selection.json`) is kept; its parameters are re-estimated per crop.
  - TFT: a fresh model with the same hyperparameters and seed, trained on every window whose decoder ends by the refit date. No data is held back, so there is no early stopping; it trains for the epoch count of the validation-selected checkpoint (`tft_epochs: null`, 2 in this run).
  - The Min-Max scaler stays fitted on the training split, as in `prepare`.
- **`train_only`**. The tables written by the `sarimax` and `tft` stages: parameters estimated once on the training split and re-applied at every validation and test origin. TFT's validation scores are not out of sample, because validation loss picked its checkpoint.

| Fold | Refit date (last day of data used) | Origins |
|------|-----------------------------------|---------|
| 0 | 2025-08-17 | 2025-08-17 to 2025-09-15 (30) |
| 1 | 2025-09-16 | 2025-09-16 to 2025-10-15 (30) |
| 2 | 2025-10-16 | 2025-10-16 to 2025-11-01 (17) |

## Metrics

Scored only on rows with `price_observed == 1`, so closed-market days are never scored. Point forecast is q50.

| Column | Definition |
|--------|------------|
| `mae`, `rmse` | of q50, LKR/kg |
| `mape_pct` | mean \|y − q50\| / y, in percent |
| `pinball_q10/q50/q90` | mean pinball loss per quantile; `quantile_loss` is their mean |
| `picp` | share of quotes inside [q10, q90]; `picp_nominal` = 0.80 |
| `interval_width` | mean q90 − q10, LKR/kg |

`metrics.csv` pools all rows of a protocol, split, and model (`crop == "all"`) and also reports each crop. `metrics_by_horizon.csv` splits the same metrics into 1–15, 16–30, 31–45, and 46–60 days ahead.

## Diebold–Mariano test

H0: TFT and SARIMAX have equal expected loss. Two-sided, alpha = 0.05.

- **Losses:** absolute error of q50, and pinball loss averaged over q10/q50/q90.
- **Series:** one loss per forecast origin, the mean over that forecast's quoted target days. Pooled (`crop == "all"`) averages the three crops per origin.
- **Variance:** consecutive origins share up to 59 target days, so the long-run variance uses h − 1 = 59 autocovariances (Diebold and Mariano, 1995). If that estimate is not positive, Bartlett weights are used instead (`variance_estimator`).
- **Small-sample correction:** Harvey, Leybourne and Newbold (1997), with a t(n − 1) reference. With n = 77 origins and h = 60 the correction multiplies the statistic by 0.23. `dm_stat` (uncorrected) and `hln_stat` (corrected, used for the p-value) are both in `diebold_mariano.csv`.

## Outputs

| File | Content |
|------|---------|
| `artifacts/forecasts/{sarimax,tft}_walkforward_forecasts.csv` | Walk-forward forecast tables, same schema as the other forecast tables |
| `artifacts/forecasts/walk_forward_folds.json` | Fold dates, SARIMAX parameters and convergence per crop and fold, TFT epochs and window counts |
| `artifacts/evaluation/metrics.csv`, `metrics_by_horizon.csv` | Metric tables |
| `artifacts/evaluation/diebold_mariano.csv` | DM results per protocol, split, crop, and loss |
| `artifacts/evaluation/plots/forecast_{crop}.png` | Observed quotes and 60-day forecasts from origins 60 days apart, one panel per model |
| `artifacts/evaluation/plots/horizon_profile.png` | MAE and q10–q90 coverage by days ahead |
| `artifacts/evaluation/summary.md` | The tables above in one page |

## First run

Walk-forward over the test split (2025-08-18 to 2025-12-31), all crops pooled, 9,135 scored rows per model:

| Model | MAE | RMSE | MAPE % | Quantile loss | PICP (80%) | Interval width |
|-------|-----|------|--------|---------------|------------|----------------|
| SARIMAX | 74.4 | 117.4 | 44.1 | 24.26 | 0.79 | 248 |
| TFT | 61.9 | 107.0 | 27.6 | 20.07 | 0.73 | 125 |

- TFT has the lower mean loss for every crop under walk-forward. Its intervals are half as wide, but they cover only 73% against the nominal 80%, falling to about 55% at 60 days ahead. SARIMAX stays near 80% at every horizon.
- SARIMAX has the lower MAE for the first 7 days ahead (25 vs 53 LKR/kg on day 1); TFT is lower from day 8 to day 60.
- **No DM test rejects H0 at 0.05**, in either protocol, for any crop or loss. Pooled walk-forward: absolute error HLN = −0.68, p = 0.50; pinball HLN = −0.45, p = 0.65. The smallest p-value is cabbage absolute error (p = 0.15).
- The uncorrected statistics are much larger (pooled walk-forward absolute error −3.0) and would reject against a normal reference. They are not used. The 77 overlapping 60-day forecasts span only 136 days, about two independent 60-day periods, so the test has little power on this test split. A difference this size cannot be called significant on these data.
- Refitting changed TFT's carrot results the most (MAE 41.7 train-only vs 61.9 walk-forward, PICP 0.96 vs 0.65). With only 2 epochs per refit and no early stopping, TFT results vary from fold to fold. SARIMAX barely changed with refitting, and all 9 SARIMAX refits converged.
