# SARIMAX Baseline (Placeholder)

## Purpose

Provide a classical **Seasonal ARIMA with eXogenous regressors (SARIMAX)** probabilistic baseline for 60-day crop price forecasting. SARIMAX serves as the interpretable statistical comparator against Temporal Fusion Transformers in this research.

## Expected Inputs

- Chronologically ordered univariate (or lightly multivariate) price series per crop/market entity
- Optional exogenous covariates (weather, calendar indicators) aligned to the same frequency
- Configuration for seasonal order, trend, and forecast horizon (`horizon: 60` in `configs/config.yaml`)

## Expected Outputs

- Multi-step (60-day) point forecasts
- Prediction intervals / distributional summaries suitable for PICP and related metrics
- Serialized model artifacts and fold-level forecast files for walk-forward evaluation

## Future Implementation

1. Stationarity checks and automatic/order-selection workflow (e.g., AIC-guided).
2. Fit SARIMAX per entity (or pooled specification if justified).
3. Produce out-of-sample forecasts under walk-forward origins.
4. Export forecasts in a schema compatible with `src/evaluation`.

**Status:** Not implemented in the current milestone. Do not add training code here until the modeling milestone.
