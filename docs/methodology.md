# Research Methodology

High-level methodology for the proposal milestone. This document describes **planned** methods; no experiments have been executed yet.

## Problem Setting

- **Domain:** Continuous-cycle cool-climate crop wholesale prices in Sri Lanka
- **Task:** Multi-step **probabilistic** price forecasting
- **Horizon:** 60 days
- **Comparators:** Temporal Fusion Transformer (TFT) vs SARIMAX baseline

## Data

Planned data assets include historical price series and aligned external covariates (weather and/or calendar signals). Datasets will be stored under `data/` and excluded from git.

## Preprocessing

1. Standardize timestamps and entity keys (crop, market).
2. Remove duplicates and resolve conflicting same-day reports.
3. Merge exogenous series on the shared calendar.
4. Impute missing values with time-aware strategies.
5. Synchronize to a regular daily frequency.

## Feature Engineering

Construct lagged prices, rolling statistics, and temporal indicators. Fit any scaler exclusively on training windows, then transform validation and test folds.

## Modeling

### SARIMAX

Classical seasonal ARIMA with exogenous regressors, producing multi-step forecasts and prediction intervals as the statistical baseline.

### TFT

Attention-based multi-horizon model (PyTorch Forecasting + Lightning) producing quantile forecasts over the 60-day horizon, conditioned on static and time-varying inputs.

## Evaluation Protocol

- **Splits:** Chronological train / validation / test ratios from `configs/config.yaml`
- **Primary protocol:** Walk-forward (rolling or expanding origin) validation
- **Point metrics:** MAE, RMSE, MAPE
- **Probabilistic metrics:** Quantile (pinball) loss, PICP
- **Significance:** Diebold–Mariano test on paired loss differentials between TFT and SARIMAX

## Reproducibility

- Fixed `random_seed` in configuration
- Documented dependency list (`requirements.txt`, `pyproject.toml`)
- Scripted pipeline under `src/` (notebooks reserved for exploration only)

## Out of Scope (Current Milestone)

- Model implementation and training
- Dataset download or synthetic data generation
- Notebook creation
- Empirical results reporting
