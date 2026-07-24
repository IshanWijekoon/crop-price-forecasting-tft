# Temporal Fusion Transformer (Placeholder)

## Purpose

Implement a **Temporal Fusion Transformer (TFT)** for 60-day **probabilistic** price forecasting of continuous-cycle cool-climate crops. TFT is the primary deep-learning model under evaluation, expected to leverage static metadata, known future inputs, and observed time-varying covariates via attention and gating mechanisms.

## Expected Inputs

- Entity-indexed panels (crop / market) with a regular time index
- Past-observed covariates, known-future inputs (e.g., calendar), and static features
- Look-back window and horizon from `configs/config.yaml` (`window_size`, `horizon: 60`)
- Quantile levels for probabilistic forecasts (to be defined in config)

## Expected Outputs

- Multi-horizon quantile forecasts (e.g., P10 / P50 / P90)
- Optional attention / variable-importance diagnostics for interpretability analysis
- Checkpoints and prediction files consumable by `src/evaluation`

## Future Implementation

1. Define PyTorch Forecasting `TimeSeriesDataSet` schemas.
2. Train TFT with Lightning under chronological splits / walk-forward folds.
3. Tune architecture and learning rate using the validation split.
4. Export quantile forecasts aligned with SARIMAX evaluation schema.
5. Compare against SARIMAX using MAE/RMSE/MAPE, quantile loss, PICP, and Diebold–Mariano tests.

**Status:** Not implemented in the current milestone. Do not add training code here until the modeling milestone.

**Planned stack:** PyTorch, PyTorch Forecasting, Lightning.
