from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from fuel_fair_price.models.brent import brent_usd_bbl_to_eur_l

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
BRENT_SERIES = "MCOILBRENTEU"
EURUSD_SERIES = "EXUSEU"


def fetch_fred_series(series_id: str, timeout: int = 30) -> pd.DataFrame:
    """Fetch a FRED series as DATE/value columns without requiring an API key."""
    url = FRED_CSV.format(series_id=series_id)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    value_col = next(c for c in df.columns if c != "DATE")
    df = df.rename(columns={value_col: "value"})
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).sort_values("DATE").reset_index(drop=True)


def fetch_brent_and_fx(timeout: int = 30) -> pd.DataFrame:
    """Monthly Brent, EUR/USD and Brent crude-equivalent cost in EUR/L."""
    brent = fetch_fred_series(BRENT_SERIES, timeout=timeout).rename(
        columns={"DATE": "date", "value": "brent_usd_bbl"}
    )
    fx = fetch_fred_series(EURUSD_SERIES, timeout=timeout).rename(
        columns={"DATE": "date", "value": "eurusd_usd_per_eur"}
    )
    out = brent.merge(fx, on="date", how="inner")
    out["brent_eur_l"] = [
        brent_usd_bbl_to_eur_l(b, fx_)
        for b, fx_ in zip(out["brent_usd_bbl"], out["eurusd_usd_per_eur"])
    ]
    return out
