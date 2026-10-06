# SARIMAX Baseline

Classical probabilistic baseline for the 60-day forecast: one **Seasonal ARIMA with exogenous regressors** per crop, fitted with statsmodels. Code: `model.py`.

```bash
python -m src.main --stages sarimax
```

## Data

- **Endogenous:** `log(price_lkr_per_kg)` on the daily calendar (`sarimax.log_target`). Days without a Dambulla quote stay missing; the Kalman filter skips them, so no price is filled in.
- **Exogenous:** the `*_lag_60` copies of `sarimax.exog` built by the `prepare` stage (diesel, USD/LKR, 30-day rainfall sums and mean temperature for both districts), Min-Max scaled on training rows. For a target day up to 60 days ahead, each value comes from the forecast origin or earlier, so it is known when the forecast is made.

## Order selection (training rows only)

1. **Differencing.** ADF test on the training log quotes. `d = 0` if a unit root is rejected at `sarimax.stationarity_alpha` (0.05), otherwise `d = 1`. AIC is not compared across different `d`.
2. **Grid search.** With `d` fixed, every combination of `p ∈ {0,1,2}`, `q ∈ {0,1}`, `P ∈ {0,1}`, `D = 0`, `Q ∈ {0,1}`, seasonal period 7, is fitted. The lowest AIC among converged fits is kept. A constant is included only when there is no differencing.

## Forecasts

Parameters are estimated once on the training split. For each validation and test window, the fitted parameters are re-applied (`results.apply`) to the history up to the origin, including validation/test quotes already known by then, and `get_forecast` projects 60 days with the window's exogenous rows. Quantiles are `mean + z_q · se` on the log scale, exponentiated, which gives log-normal price quantiles (q10, q50, q90).

The `walkforward` stage (`src/evaluation/walk_forward_validation.py`) keeps the selected order, re-estimates the parameters every 30 days on an expanding window over the test split, and writes `artifacts/forecasts/sarimax_walkforward_forecasts.csv`.

## Outputs

| File | Content |
|------|---------|
| `artifacts/forecasts/sarimax_forecasts.csv` | Forecast table, schema in `src/models/forecast_output.py` |
| `artifacts/models/sarimax/selection.json` | Per crop: ADF p-value, chosen `d`, every grid candidate with AIC and convergence, selected order, fitted parameters |

## First run

| Crop | ADF p | Selected model |
|------|-------|----------------|
| Beans | 0.004 | ARMA(1,1) with constant, no seasonal term |
| Cabbage | 0.226 | ARIMA(1,1,1) |
| Carrot | 0.434 | ARIMA(1,1,1)(1,0,1)₇ |
