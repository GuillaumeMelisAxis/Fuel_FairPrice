from __future__ import annotations

from io import StringIO
import json
import pandas as pd
import requests

from fuel_fair_price.config import (
    MAINLAND_EX_CORSICA_REGION_CODES,
    STATION_EXPORT_URL,
)

KEEP_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "cp",
    "pop",
    "adresse",
    "ville",
    "departement",
    "code_departement",
    "region",
    "code_region",
    "gazole_maj",
    "gazole_prix",
    "sp95_maj",
    "sp95_prix",
    "gazole_rupture_type",
    "sp95_rupture_type",
]

MAX_PRICE_AGE_HOURS = 72


def fetch_current_stations(timeout: int = 60) -> pd.DataFrame:
    """Download the official DGCCRF real-time station feed."""
    response = requests.get(STATION_EXPORT_URL, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep=";", dtype={"cp": "string"})


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep SP95/Gazole observations for mainland metropolitan France ex Corsica."""
    out = df.copy()

    existing = [c for c in KEEP_COLUMNS if c in out.columns]
    out = out[existing]

    out["code_region"] = (
        pd.to_numeric(out["code_region"], errors="coerce")
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )

    out = out[
        out["code_region"].isin(MAINLAND_EX_CORSICA_REGION_CODES)
    ]

    for col in ["sp95_prix", "gazole_prix"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["sp95_maj", "gazole_maj"]:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)

    # A = autoroute, R = route in the official source.
    if "pop" in out:
        out["road_type"] = out["pop"].map({"A": "AUTOROUTE", "R": "ROUTE"}).fillna("UNKNOWN")

    return out.reset_index(drop=True)

def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["code_region"] = (
        pd.to_numeric(
            out["code_region"],
            errors="coerce",
        )
        .astype("Int64")
        .astype("string")
        .str.zfill(2)
    )

    out = out[
        out["code_region"].isin(
            MAINLAND_EX_CORSICA_REGION_CODES
        )
    ].copy()

    return out

def _parse_raw_prices(value):
    """Parse the raw `prix` field returned by the official dataset."""
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


def _extract_fuel(raw_prices, fuel_name):
    """
    Extract price and update timestamp for one fuel
    from the raw `prix` field.
    """
    prices = _parse_raw_prices(raw_prices)

    for item in prices:
        if item.get("@nom") != fuel_name:
            continue

        price = pd.to_numeric(
            item.get("@valeur"),
            errors="coerce",
        )

        raw_date = item.get("@maj")

        if not raw_date:
            updated_at = pd.NaT
        else:
            updated_at = pd.Timestamp(raw_date)

            # @maj is expressed in French local time
            if updated_at.tzinfo is None:
                updated_at = updated_at.tz_localize(
                    "Europe/Paris",
                    ambiguous="NaT",
                    nonexistent="NaT",
                )

        return price, updated_at

    return float("nan"), pd.NaT

def to_long_format(df: pd.DataFrame) -> pd.DataFrame:

    if "prix" not in df.columns:
        raise KeyError(
            "Column 'prix' is required to reconstruct "
            "raw fuel prices and timestamps."
        )

    common_cols = [
        "id",
        "ville",
        "code_region",
    ]

    common_cols = [
        col for col in common_cols
        if col in df.columns
    ]

    fuel_mapping = {
        "SP95": "SP95",
        "GAZOLE": "Gazole",
    }

    frames = []

    for fuel, raw_name in fuel_mapping.items():

        tmp = df[common_cols + ["prix"]].copy()

        extracted = tmp["prix"].apply(
            lambda value: _extract_fuel(
                value,
                raw_name,
            )
        )

        tmp["price_eur_l"] = extracted.apply(
            lambda x: x[0]
        )

        tmp["updated_at"] = extracted.apply(
            lambda x: x[1]
        )

        tmp["fuel"] = fuel

        tmp = tmp.drop(
            columns=["prix"]
        )

        frames.append(tmp)

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    # Remove stations that do not sell this fuel
    out = out.dropna(
        subset=[
            "price_eur_l",
            "updated_at",
        ]
    ).copy()

    # Everything remains in Europe/Paris
    now = pd.Timestamp.now(
        tz="Europe/Paris"
    )

    out["age_hours"] = (
        now - out["updated_at"]
    ).dt.total_seconds() / 3600

    out["price_age_days"] = (
        out["age_hours"] / 24
    )

    out["freshness_flag"] = "CURRENT"

    out.loc[
        out["price_age_days"] > 7,
        "freshness_flag",
    ] = "OLD"

    out.loc[
        out["price_age_days"] > 30,
        "freshness_flag",
    ] = "VERY_OLD"

    return out.reset_index(drop=True)