# Observed Dataset

How the modeling data in `data/processed/` is built, what it contains, and which raw files were left out and why.

Rebuild everything with:

```bash
python -m src.main --stages build
```

The build reads `configs/config.yaml` (section `dataset`) and writes four files. All of them are gitignored and regenerated from `data/raw/`.

| File | Grain | Content |
|------|-------|---------|
| `price_observations.csv` | one row per quote | Every real price quote used or checked: CBSL daily (Dambulla, Pettah, Narahenpita; wholesale and retail) and WFP monthly market prices |
| `covariates_daily.csv` | district x day, 2018-01-01 to 2025-12-31 | Weather, dekadal rainfall, fuel, USD/LKR, market-open flag, calendar |
| `modeling_panel.csv` | crop x calendar day, 2023-05-15 to 2025-12-31 | Target, price lags and rolling stats, both districts' weather as columns, national covariates, split label |
| `build_report.json` | — | Coverage per source, anomaly decisions, split boundaries, empty-cell counts |

No price is generated, interpolated, or modelled. Days without a quote keep a missing target.

## Target

**Daily wholesale price at the Dambulla Dedicated Economic Centre, LKR/kg, for beans, cabbage, and carrot.** Source: Central Bank of Sri Lanka, Economic Data Library, "Daily Commodity Prices – Wholesale Prices of Vegetables" (`data/raw/Data.xls`, an HTML table export, last updated 2026-04-21).

Dambulla is the main wholesale hub for up-country vegetables, so it is used as the market price for Nuwara Eliya and Badulla produce. The CBSL Dambulla series tracks WFP's observed district retail prices closely (monthly means, 2023-06 to 2025-09, 28 months):

| Crop | WFP Nuwara Eliya retail | WFP Badulla retail | Retail / Dambulla wholesale |
|------|------------------------|--------------------|-----------------------------|
| Carrot | r = 0.99 | r = 0.96 | 1.5x |
| Cabbage | r = 0.98 | r = 0.96 | 1.8x |
| Beans | r = 0.94 | r = 0.80 | 1.4x |

The Pettah wholesale series is kept in `price_observations.csv` and used only to confirm or reject large one-day jumps at Dambulla (see Anomaly checks).

### Horizon on a daily target

The 60-day horizon is 60 calendar days. Dambulla quotes exist on weekdays that are not public or Poya holidays, so a 60-day window holds 34 to 44 quoted days (mean 39.3). The target is missing on the other days. Models should be scored only on rows with `price_observed == 1`.

## Covariates

| Variable | Source | Treatment |
|----------|--------|-----------|
| Rainfall, tmin, tmax, tmean, ET0 (daily) | Open-Meteo historical reanalysis at the Nuwara Eliya and Badulla market coordinates (`sources/weather_daily.csv`) | Implausible values set to missing (none found), then time interpolation and edge fill per district. `weather_imputed` flags filled days (0 in this build). |
| Rainfall rolling sums, 7 and 30 days | Built from daily rainfall | Previous days only, today excluded. Empty for the first 7/30 days of 2018; those are the 74 empty cells in `covariates_daily.csv`, all before the panel starts. |
| Dekadal rainfall (`rfh`) and anomaly (`rfq`, % of long-term mean) | HDX / WFP CHIRPS subnational rainfall (`lka-rainfall-subnat-full.csv`), district codes LK23 Nuwara Eliya, LK81 Badulla | Each dekad is attached from the day after it ends, so only completed dekads are visible. Rows with `version == forecast` are dropped. |
| Auto diesel, petrol 92 (LKR/L) | Ceylon Petroleum Corporation revision table (`sources/fuel_revisions_cpc.csv`) | Step series carried forward between revisions. `fuel_revision_day` marks revision dates. Cross-checked against `sources/fuel_daily.csv` with no differences. |
| USD/LKR | Yahoo Finance `LKR=X` daily close (`sources/usd_lkr_daily.csv`) | Forward-filled on days with no print. `usd_lkr_carried_forward` marks those days (868 of 2,922). |
| `market_open` | CBSL Dambulla quote present on the day | 1 if any study crop was quoted. Before 2023-05-15 (covariates file only) it falls back to a Monday–Friday rule; `market_open_source` says which. Closures match the public holiday and Poya calendar, which is published in advance, so the flag is treated as known-future. |
| Calendar | Date | Year, month, day of week, day of year, ISO week, month sine/cosine, monsoon season (northeast Dec–Feb, first inter-monsoon Mar–Apr, southwest May–Sep, second inter-monsoon Oct–Nov). |

## Panel columns by role

| Role | Columns |
|------|---------|
| Time index / group | `date`, `time_idx` (days since 2023-05-15), `crop` |
| Static | `crop`, `market`, `price_type` |
| Target | `price_lkr_per_kg` (missing on closed days) |
| Target support | `price_observed`, `price_ffill` (last quote, for lag building only), `price_is_carried_forward` |
| Known future | `market_open`, `year`, `month`, `day_of_week`, `day_of_year`, `week_of_year`, `month_sin`, `month_cos`, `season` |
| Observed past (price) | `price_lag_{1,7,14,30,60}` = last quote at or before t−k days; `price_roll_{mean,std}_{5,22}q` = previous 5 / 22 quotes, today excluded |
| Observed past (exogenous) | `diesel_lkr_per_litre`, `petrol92_lkr_per_litre`, `fuel_revision_day`, `usd_lkr`, `usd_lkr_carried_forward`, and per district `{nuwara_eliya,badulla}_{rainfall_mm, tmin_c, tmax_c, tmean_c, et0_mm, rainfall_roll_sum_7, rainfall_roll_sum_30, rain_dekad_mm, rain_dekad_anomaly_pct, weather_imputed}` |
| Split | `split` ∈ {warmup, train, validation, test} |

The weather of both districts sits on the same row as one national market price. The price is not duplicated per district.

Price rolling windows count quotes, not calendar days, because the April Sinhala and Tamil New Year closure lasts up to a week and would empty a 7-calendar-day window.

## Preprocessing steps

1. **Dates.** Parsed with explicit formats, normalised to midnight, sorted. Unparseable rows are dropped and logged (none).
2. **Duplicates.** Exact duplicates dropped; rows that share keys (date, market, crop, price type) but differ are averaged. None found in any source.
3. **Invalid values.** Non-positive prices set to missing (none). Weather plausibility bounds as above.
4. **Covariate imputation.** Weather: time interpolation per district. Fuel and FX: forward fill, flagged. Target: never imputed.
5. **Alignment.** Every series is reindexed onto one daily calendar; district covariates are pivoted to columns and joined on date.
6. **Features.** Lags, quote-based rolling stats, calendar and monsoon features.
7. **Scaling.** Not applied in `modeling_panel.csv`. The `prepare` stage (below) fits Min-Max on training rows only and reuses that scaler for validation and test.

### Anomaly checks

A Dambulla quote is removed from the target only if all of these hold:

- the log change into the quote and the log change out of it both exceed 6 robust z-scores (median / MAD of daily log changes),
- the two changes have opposite signs (a one-day spike that reverts),
- Pettah wholesale on the same day does not move the same way by at least a quarter of the size.

Two large jumps were found and both were kept because Pettah wholesale and Dambulla retail show the same move:

- Beans, 2024-04-17, 45 LKR/kg (165 the day before): post-New Year glut; Pettah fell from 130 to 80.
- Carrot, 2025-12-01, 850 LKR/kg (230 on the previous quote): first trading day after the late-November 2025 cyclone and floods; Pettah rose from 280 to 600.

No quote was removed in this build. Flagged quotes, if any, keep `anomaly_flag = 1` in `price_observations.csv` and are listed in `build_report.json`.

## Splits

Chronological, shared by all three crops. The first 60 days are warm-up because `price_lag_60` needs 60 days of history.

| Split | From | To | Days | Observed target rows (3 crops) |
|-------|------|----|------|-------------------------------|
| Warm-up | 2023-05-15 | 2023-07-13 | 60 | — |
| Train (70%) | 2023-07-14 | 2025-04-04 | 631 | 1,248 |
| Validation (15%) | 2025-04-05 | 2025-08-17 | 135 | 251 |
| Test (15%) | 2025-08-18 | 2025-12-31 | 136 | 273 |

## Excluded files

| File | Reason |
|------|--------|
| `raw/sri_lanka_crop_prices.csv` | **Suspected synthetic; not used.** It is labelled as monthly farm and retail prices 2000–2025, but: month-on-month changes are uncorrelated across all nine crops (|r| ≤ 0.2), while real Sri Lankan vegetable prices move together; prices rise smoothly by about 5% a year with no 2022 crisis jump; the farm/retail ratio stays inside 0.69–0.80 for 25 years; and it does not track observed prices. Correlations with WFP district retail run from −0.31 to +0.27, and its "Dambulla retail" cabbage averages 232 LKR/kg in 2020–21 against WFP's observed Dambulla wholesale of 63. |
| `raw/integrated/*` | Wholesale price column is generated (`wholesale_price_origin = synthetic_dgp_v1`, see `raw/DATASET.md`). Its weather, fuel, and FX inputs are the same files in `raw/sources/`, which are used directly. |
| `raw/sl_fuel_prices_monthly_2010_2025.csv` | File notes say "Synthetic monthly price index; not real market". CPC revisions are used instead. |
| CBSL quotes after 2025-12-31 | Kept in `price_observations.csv`, but weather and FX files stop at 2025-12-31, so the panel ends there. |
| WFP prices as a target | District series are monthly and mostly 2023-06 to 2025-09 (28 points); wholesale rows are only 2020–2021 at Dambulla and Pettah. Kept in `price_observations.csv` as a validation check. WFP "Beans (mung)" is a pulse, not green beans, and is not loaded. |
| Leeks, beetroot | No observed price series in the collected files. Local potato exists only as WFP monthly retail and is kept in `price_observations.csv` only. |
| `raw/climate-change_lka.csv` | Annual national indicators; no information at a 60-day horizon. |
| `raw/SriLanka_Weather_Dataset*.csv`, `raw/weatherData.csv`, `raw/locationData.csv` | City weather dumps that omit Nuwara Eliya or stop before 2025. The Open-Meteo district series covers both districts through 2025. |
| `raw/EXSLUS.csv` | FRED monthly average USD/LKR; the daily series is used instead. |
| Farmer and market-officer surveys | Not collected. No survey rows were created. |

## Limitations

- **Short history.** 2.6 years of daily quotes (about 630 trading days per crop). A yearly seasonal term cannot be estimated from one and a half training years; weekly and monsoon effects are what the data can support.
- **One national market.** Dambulla is a proxy for Nuwara Eliya and Badulla farm-gate conditions. District weather enters as covariates; there are no district wholesale prices.
- **Shock in the test period.** The test split includes the late-November 2025 cyclone price spike, which is outside anything seen in training for carrot and cabbage at that scale.
- **Reanalysis weather.** Open-Meteo values are gridded reanalysis, not station observations.

## Model inputs (`prepare` stage)

```bash
python -m src.main --stages prepare
```

`src/features/prepare_model_data.py` reads `modeling_panel.csv` and the `features`, `forecast`, and `sarimax` sections of `configs/config.yaml`, and writes three more gitignored files:

| File | Content |
|------|---------|
| `model_frame.csv` | Panel without warm-up rows (2,706 rows = 3 crops x 902 days). Predictors Min-Max scaled; target, flags, and categoricals left raw. |
| `forecast_windows.csv` | One row per crop and forecast origin: `encoder_start`, `origin`, `decoder_start`, `decoder_end`, `split`, `observed_targets`. |
| `minmax_scaler.json` | Training min and max per scaled column, and the training period they were fitted on. |

Steps:

1. **Horizon-lagged SARIMAX exogenous inputs.** Each column in `sarimax.exog` gets a copy shifted by `sarimax.exog_lag_days` (60) days, named `{column}_lag_60`. For any decoder day t+h with h ≤ 60, that value comes from day t or earlier, so it is known when the forecast is made. The shift is taken before the warm-up is dropped, so the first training day uses the first warm-up day. TFT gets the same information by seeing those covariates only in its encoder.
2. **Warm-up dropped.** The 60 warm-up days exist only to fill `price_lag_60`. After dropping them, no predictor has an empty cell; the build fails if one does.
3. **Min-Max on training rows only.** `known_reals`, `observed_reals`, and the lagged SARIMAX inputs are scaled to [0, 1] using training rows (2023-07-14 to 2025-04-04). Validation and test use that scaler as is, so values can fall outside [0, 1]. In this build, test rainfall reaches 2.38 for Nuwara Eliya daily rainfall because of the November 2025 cyclone. The target `price_lkr_per_kg`, `price_ffill`, the 0/1 flags, `crop`, and `season` are not scaled.
4. **Sliding windows.** Encoder = 90 calendar days (`forecast.window_size`) ending at the origin; decoder = the next 60 days (`forecast.horizon`); origins one day apart (`forecast.origin_step`). A window is labelled with the split that contains its whole decoder; windows whose decoder crosses a split boundary are dropped. The encoder may reach back into earlier splits, because those days are already known at the origin.

| Split | Windows per crop | Origins |
|-------|------------------|---------|
| Train | 482 | 2023-10-11 to 2025-02-03 |
| Validation | 76 | 2025-04-04 to 2025-06-18 |
| Test | 77 | 2025-08-17 to 2025-11-01 |

### Column roles in `configs/config.yaml` (`features`)

| Role | Columns |
|------|---------|
| Static categorical | `crop` |
| Known-future categorical | `season` |
| Known-future real (scaled) | `month_sin`, `month_cos`, `day_of_week`, `day_of_year` |
| Known-future flag | `market_open` |
| Observed real (scaled) | price lags 1/7/14/30/60, quote-based rolling mean/std 5q/22q, diesel, petrol 92, USD/LKR, and for both districts: daily rainfall, mean temperature, ET0, 30-day rainfall sum, dekadal rainfall anomaly |
| Observed flag | `price_is_carried_forward`, `fuel_revision_day`, `usd_lkr_carried_forward` |
| Target / scoring mask | `price_lkr_per_kg` / `price_observed` |

Left out of the model roles: `year` and `week_of_year`, which repeat information already in the other calendar columns; tmin, tmax, the 7-day rainfall sum, and dekadal rainfall totals, which overlap strongly with the columns kept; and `weather_imputed`, which is 0 on every row.

### Settings that differ from the monthly plan

The original plan assumed a monthly target: horizon 2 months, encoder 12 months, SARIMAX seasonal period 12. With a daily target, the horizon is 60 calendar days, the encoder is 90 days, and the SARIMAX order grid uses a weekly seasonal period of 7. A yearly SARIMAX season cannot be estimated from one and a half years of training data.
