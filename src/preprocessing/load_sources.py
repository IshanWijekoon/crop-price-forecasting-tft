from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.preprocessing.clean_data import clean_data, remove_duplicates, standardize_dates

logger = logging.getLogger(__name__)

PRICE_COLUMNS = [
    "date",
    "source",
    "market",
    "admin1",
    "admin2",
    "crop",
    "price_type",
    "unit",
    "price_lkr_per_kg",
    "series_name",
]

CBSL_CROP_NAMES = {"Beans": "beans", "Cabbage": "cabbage", "Carrot": "carrot"}


def load_cbsl_daily_prices(path: str | Path) -> pd.DataFrame:
    """Parse the CBSL Economic Data Library export (an HTML table saved with an .xls name).

    Item names look like `Vegetable-Dambulla-Wholesale-Beans`. Days with no quote
    (weekends, holidays) are blank in the export and are not returned.
    """
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html))
    grid = max(tables, key=lambda t: t.size)

    grid = grid.dropna(subset=["Unit"]).drop(columns=[c for c in grid.columns if str(c).startswith("Unnamed")])
    long = grid.melt(id_vars=["Item Name", "Unit", "Scale"], var_name="date", value_name="price_lkr_per_kg")
    long["price_lkr_per_kg"] = pd.to_numeric(long["price_lkr_per_kg"], errors="coerce")
    long = long.dropna(subset=["price_lkr_per_kg"])

    parts = long["Item Name"].str.split("-", n=3, expand=True)
    long["market"] = parts[1].str.strip()
    long["price_type"] = parts[2].str.strip().str.lower()
    long["crop"] = parts[3].str.strip().map(CBSL_CROP_NAMES)
    long = long.dropna(subset=["crop"])

    scale = long["Scale"].str.lower().str.replace(" ", "")
    if not scale.eq("perkg").all():
        raise ValueError(f"Unexpected CBSL scale values: {long['Scale'].unique()}")

    out = pd.DataFrame(
        {
            "date": long["date"],
            "source": "CBSL",
            "market": long["market"],
            "admin1": pd.NA,
            "admin2": pd.NA,
            "crop": long["crop"],
            "price_type": long["price_type"],
            "unit": "LKR/kg",
            "price_lkr_per_kg": long["price_lkr_per_kg"],
            "series_name": long["Item Name"],
        }
    )
    out = clean_data(
        out,
        keys=["date", "market", "crop", "price_type"],
        value_columns=["price_lkr_per_kg"],
        date_format="%Y-%m-%d",
    )
    logger.info("CBSL: %d quotes, %s to %s", len(out), out["date"].min().date(), out["date"].max().date())
    return out[PRICE_COLUMNS]


def load_wfp_prices(path: str | Path, commodity_map: Mapping[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[~raw["date"].astype(str).str.startswith("#")]
    raw = raw[raw["commodity"].isin(commodity_map) & raw["unit"].eq("KG") & raw["currency"].eq("LKR")]

    out = pd.DataFrame(
        {
            "date": raw["date"],
            "source": "WFP",
            "market": raw["market"],
            "admin1": raw["admin1"],
            "admin2": raw["admin2"],
            "crop": raw["commodity"].map(commodity_map),
            "price_type": raw["pricetype"].str.lower(),
            "unit": "LKR/kg",
            "price_lkr_per_kg": pd.to_numeric(raw["price"], errors="coerce"),
            "series_name": raw["commodity"],
        }
    )
    out = clean_data(
        out,
        keys=["date", "market", "crop", "price_type"],
        value_columns=["price_lkr_per_kg"],
        date_format="%m/%d/%Y",
    )
    logger.info("WFP: %d quotes, %s to %s", len(out), out["date"].min().date(), out["date"].max().date())
    return out[PRICE_COLUMNS]


WEATHER_COLUMNS = ["rainfall_mm", "tmin_c", "tmax_c", "tmean_c", "et0_mm"]


def load_weather_daily(path: str | Path) -> pd.DataFrame:
    """Open-Meteo daily reanalysis per district. Physically impossible values become missing."""
    raw = pd.read_csv(path)
    df = clean_data(raw, keys=["date", "district_key"], value_columns=[], date_format="%Y-%m-%d")

    bad = pd.DataFrame(index=df.index)
    bad["rainfall_mm"] = (df["rainfall_mm"] < 0) | (df["rainfall_mm"] > 500)
    bad["et0_mm"] = (df["et0_mm"] < 0) | (df["et0_mm"] > 15)
    for col in ["tmin_c", "tmax_c", "tmean_c"]:
        bad[col] = (df[col] < -5) | (df[col] > 45)
    inverted = df["tmin_c"] > df["tmax_c"]
    for col in ["tmin_c", "tmax_c"]:
        bad[col] |= inverted

    for col in WEATHER_COLUMNS:
        n = int(bad[col].sum())
        if n:
            logger.warning("Weather: %d implausible %s values set to missing", n, col)
            df.loc[bad[col], col] = np.nan
    return df[["date", "district_key", *WEATHER_COLUMNS]]


def load_dekadal_rainfall(path: str | Path, pcode_map: Mapping[str, str]) -> pd.DataFrame:
    """HDX/WFP CHIRPS dekadal rainfall per district.

    `available_from` is the day after the dekad ends, so a dekad's total is only
    used once it is complete.
    """
    raw = pd.read_csv(path)
    raw = raw[raw["PCODE"].isin(pcode_map) & raw["version"].ne("forecast")]
    df = standardize_dates(raw, date_format="%Y-%m-%d")
    df["district_key"] = df["PCODE"].map(pcode_map)
    df = remove_duplicates(df, keys=["district_key", "date"], value_columns=["rfh", "rfq"])

    day = df["date"].dt.day
    end = df["date"] + pd.to_timedelta(9, unit="D")
    month_end = df["date"] + pd.offsets.MonthEnd(0)
    df["dekad_end"] = end.where(day < 21, month_end)
    df["available_from"] = df["dekad_end"] + pd.Timedelta(days=1)

    return df.rename(
        columns={"date": "dekad_start", "rfh": "rain_dekad_mm", "rfq": "rain_dekad_anomaly_pct"}
    )[["district_key", "dekad_start", "dekad_end", "available_from", "rain_dekad_mm", "rain_dekad_anomaly_pct", "version"]]


def load_fuel_daily(
    revisions_path: str | Path,
    start: str,
    end: str,
    check_path: str | Path | None = None,
) -> pd.DataFrame:
    """Expand CPC retail fuel revisions to a daily step series."""
    rev = pd.read_csv(revisions_path)
    rev = clean_data(
        rev,
        keys=["date"],
        value_columns=["petrol92_lkr_per_litre", "diesel_lkr_per_litre"],
        date_format="%Y-%m-%d",
    )
    calendar = pd.date_range(rev["date"].min(), end, freq="D", name="date")
    daily = rev.set_index("date").reindex(calendar)
    daily["fuel_revision_day"] = daily["diesel_lkr_per_litre"].notna().astype("int8")
    daily[["petrol92_lkr_per_litre", "diesel_lkr_per_litre"]] = daily[
        ["petrol92_lkr_per_litre", "diesel_lkr_per_litre"]
    ].ffill()
    daily = daily.loc[start:end].reset_index()

    if check_path is not None:
        check = pd.read_csv(check_path, parse_dates=["date"]).set_index("date")
        joined = daily.set_index("date").join(check, rsuffix="_check", how="inner")
        mismatch = (
            (joined["diesel_lkr_per_litre"] != joined["diesel_lkr_per_litre_check"])
            | (joined["petrol92_lkr_per_litre"] != joined["petrol92_lkr_per_litre_check"])
        ).sum()
        if mismatch:
            logger.warning("Fuel: %d days differ from %s", int(mismatch), check_path)
    return daily


def load_fx_daily(path: str | Path, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = clean_data(raw, keys=["date"], value_columns=["usd_lkr"], date_format="%Y-%m-%d")
    calendar = pd.date_range(start, end, freq="D", name="date")
    df = df.set_index("date")[["usd_lkr"]].reindex(calendar)

    missing = df["usd_lkr"].isna()
    df["usd_lkr"] = df["usd_lkr"].ffill().bfill()
    repeated = df["usd_lkr"].diff().eq(0)
    df["usd_lkr_carried_forward"] = (missing | repeated).astype("int8")
    return df.reset_index()
