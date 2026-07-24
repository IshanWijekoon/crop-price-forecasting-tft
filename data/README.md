# Data Directory

This directory stores research datasets used by the crop price forecasting pipeline.

## Layout

| Subfolder | Purpose |
|-----------|---------|
| `raw/` | Original, immutable market price extracts as obtained from sources |
| `processed/` | Cleaned, merged, and feature-ready panels produced by `src/preprocessing` and `src/features` |
| `external/` | Auxiliary covariates (e.g., weather, calendar, macroeconomic signals) |

## Version control policy

- CSV and other large data files are **gitignored** (see repository `.gitignore`).
- Only folder structure markers (`.gitkeep`) and this README are tracked.
- Document dataset provenance, licenses, and acquisition steps in the research report; do not commit proprietary or personally sensitive files.

## Current status

No datasets are included in this milestone. Place acquired files under the appropriate subfolder when the data milestone begins.
