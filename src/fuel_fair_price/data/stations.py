from __future__ import annotations

from io import StringIO

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

    out["code_region"] = out["code_region"].astype("string").str.zfill(2)
    out = out[out["code_region"].isin(MAINLAND_EX_CORSICA_REGION_CODES)]

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


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per station and fuel, restricted to SP95 and Gazole."""
    frames: list[pd.DataFrame] = []

    mapping = {
        "SP95": ("sp95_prix", "sp95_maj", "sp95_rupture_type"),
        "GAZOLE": ("gazole_prix", "gazole_maj", "gazole_rupture_type"),
    }

    id_cols = [
        c for c in [
            "id", "latitude", "longitude", "cp", "adresse", "ville",
            "departement", "code_departement", "region", "code_region",
            "road_type",
        ] if c in df.columns
    ]

    for fuel, (price_col, update_col, rupture_col) in mapping.items():
        if price_col not in df.columns:
            continue
        cols = id_cols + [c for c in [price_col, update_col, rupture_col] if c in df.columns]
        tmp = df[cols].copy()
        tmp["fuel"] = fuel
        tmp = tmp.rename(columns={price_col: "price_eur_l", update_col: "price_updated_at"})
        if rupture_col in tmp.columns:
            tmp = tmp.rename(columns={rupture_col: "rupture_type"})
            tmp = tmp[tmp["rupture_type"].isna() | (tmp["rupture_type"].astype(str).str.len() == 0)]
        tmp = tmp.dropna(subset=["price_eur_l"])
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
