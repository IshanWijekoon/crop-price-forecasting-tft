# Temporal Fusion Transformer

Deep probabilistic model under evaluation: one global **Temporal Fusion Transformer** across the three crops, built with PyTorch Forecasting and trained with Lightning on CPU. Code: `model.py`.

```bash
python -m src.main --stages tft
```

## Data

`TimeSeriesDataSet` over `model_frame` (from `prepare`), grouped by `crop`, with column roles from `configs/config.yaml` (`features`):

| Input | Columns |
|-------|---------|
| Static | `crop`, plus the per-crop target centre and scale added by the normalizer |
| Known future (encoder and decoder) | `season`, `month_sin`, `month_cos`, `day_of_week`, `day_of_year`, `market_open` |
| Observed past (encoder only) | the carried-forward price, price lags and rolling stats, fuel, USD/LKR, district weather and rainfall, observed flags |

- **Window:** 90-day encoder, 60-day decoder, both fixed length. Training windows have their whole decoder in the train split. Validation and test windows have their whole decoder in that split, and their encoder may reach back into earlier splits, the same as `forecast_windows.csv`.
- **Target and missing quotes:** the dataset cannot hold a missing target, so the network's target is `price_ffill` (last quote carried over). `price_observed` is passed as the loss weight, so carried-over days add nothing to the training loss or to the validation loss used for early stopping.
- **Normalization:** `GroupNormalizer` per crop with a log transform (`tft.target_transformation`). Predictions are mapped back to LKR/kg before the loss, so the quantile loss is in price units.

## Model and training

Hidden size 16, 1 attention head, continuous hidden size 8, 1 LSTM layer, dropout 0.1, `QuantileLoss` at 0.1 / 0.5 / 0.9, Adam with learning rate 0.01, reduce-on-plateau after 3 epochs, gradient clip 0.1, batch size 64, at most 30 epochs, early stopping on validation loss with patience 5, seed 42. The best checkpoint by validation loss is reloaded for prediction.

Because the validation windows choose the checkpoint, only the test forecasts are fully out of sample.

## Outputs

| File | Content |
|------|---------|
| `artifacts/forecasts/tft_forecasts.csv` | Forecast table, same schema and rows as SARIMAX (`src/models/forecast_output.py`) |
| `artifacts/models/tft/tft-best.ckpt` | Best checkpoint |
| `artifacts/models/tft/logs/` | Per-epoch training and validation loss (CSV) |

Quantiles that cross are sorted per row before writing; the count is logged.

## First run

42,133 parameters. Validation loss was lowest after epoch 1 (60.7) and rose from epoch 2 while training loss kept falling, so early stopping ended the run at epoch 7 and the epoch-1 checkpoint is used. On validation windows the 10–90% interval covered only 41–63% of observed prices, against a nominal 80%. Regularisation and tuning are not part of this first pass.
