from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_taxes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["effective_from"] = pd.to_datetime(df["effective_from"])
    df["effective_to"] = pd.to_datetime(df["effective_to"], errors="coerce")
    df["fuel"] = df["fuel"].str.upper()
    return df


def get_tax_row(df: pd.DataFrame, fuel: str, as_of: str | pd.Timestamp) -> pd.Series:
    as_of = pd.Timestamp(as_of)
    fuel = fuel.upper()
    mask = (
        (df["fuel"] == fuel)
        & (df["effective_from"] <= as_of)
        & (df["effective_to"].isna() | (df["effective_to"] >= as_of))
    )
    matches = df.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one tax row for {fuel} on {as_of.date()}, got {len(matches)}")
    return matches.iloc[0]
