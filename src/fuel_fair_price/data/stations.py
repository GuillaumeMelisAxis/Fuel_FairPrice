from __future__ import annotations

from io import StringIO
import json

import pandas as pd
import requests

from fuel_fair_price.config import MAINLAND_EX_CORSICA_REGION_CODES, STATION_EXPORT_URL

PARIS_TZ = "Europe/Paris"


def fetch_current_stations(timeout: int = 60) -> pd.DataFrame:
    """Download the official DGCCRF real-time station feed."""
    response = requests.get(STATION_EXPORT_URL, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(
        StringIO(response.text),
        sep=";",
        dtype={"cp": "string", "code_region": "string", "code_departement": "string"},
    )


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Geographic cleaning only; raw ``prix`` is intentionally preserved."""
    out = df.copy()
    out["code_region"] = (
        pd.to_numeric(out["code_region"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )
    out = out[out["code_region"].isin(MAINLAND_EX_CORSICA_REGION_CODES)].copy()
    if "pop" in out.columns:
        out["road_type"] = out["pop"].map({"A": "AUTOROUTE", "R": "ROUTE"}).fillna("UNKNOWN")
    return out.reset_index(drop=True)


def _parse_raw_prices(value) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _extract_fuel(raw_prices, fuel_name: str) -> tuple[float, pd.Timestamp]:
    for item in _parse_raw_prices(raw_prices):
        if item.get("@nom") != fuel_name:
            continue
        price = pd.to_numeric(item.get("@valeur"), errors="coerce")
        raw_date = item.get("@maj")
        if not raw_date:
            return price, pd.NaT
        ts = pd.Timestamp(raw_date)
        if ts.tzinfo is None:
            ts = ts.tz_localize(PARIS_TZ, ambiguous="NaT", nonexistent="NaT")
        else:
            ts = ts.tz_convert(PARIS_TZ)
        return float(price) if pd.notna(price) else float("nan"), ts
    return float("nan"), pd.NaT


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station/fuel using the authoritative raw ``prix`` object."""
    if "prix" not in df.columns:
        raise KeyError("Column 'prix' is required to reconstruct fuel prices and timestamps")

    common_cols = [
        c for c in [
            "id", "latitude", "longitude", "cp", "adresse", "ville", "departement",
            "code_departement", "region", "code_region", "road_type",
        ] if c in df.columns
    ]
    mapping = {"SP95": "SP95", "GAZOLE": "Gazole"}
    frames = []

    for fuel, raw_name in mapping.items():
        tmp = df[common_cols + ["prix"]].copy()
        extracted = tmp["prix"].apply(lambda value: _extract_fuel(value, raw_name))
        tmp["price_eur_l"] = extracted.apply(lambda x: x[0])
        tmp["updated_at"] = extracted.apply(lambda x: x[1])
        tmp["fuel"] = fuel
        tmp = tmp.drop(columns=["prix"])
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["price_eur_l", "updated_at"]).copy()

    now = pd.Timestamp.now(tz=PARIS_TZ)
    out["age_hours_raw"] = (now - out["updated_at"]).dt.total_seconds() / 3600.0
    out["age_hours"] = out["age_hours_raw"]
    out["price_age_days"] = out["age_hours"] / 24.0
    out["freshness_flag"] = "CURRENT"
    out.loc[out["price_age_days"] > 7, "freshness_flag"] = "OLD"
    out.loc[out["price_age_days"] > 30, "freshness_flag"] = "VERY_OLD"
    return out.reset_index(drop=True)
