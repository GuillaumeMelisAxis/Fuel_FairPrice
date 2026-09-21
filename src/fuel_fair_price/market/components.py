from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "fuel",
    "refined_quote_eur_l",
    "observed_distribution_margin_eur_l",
}


def load_market_components(path: str | Path) -> pd.DataFrame:
    """Load weekly DGEC market components prepared from the official NPG bulletin."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in market components: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["fuel"] = df["fuel"].str.upper()
    return df.sort_values(["fuel", "date"]).reset_index(drop=True)


def add_normal_distribution_margin(
    df: pd.DataFrame,
    window_weeks: int = 52,
    min_periods: int = 13,
) -> pd.DataFrame:
    """
    Estimate a 'normal' transport-distribution component with a trailing median.

    A median is intentionally used instead of the current observed margin so an
    unusually high current margin does not mechanically redefine itself as fair.
    """
    out = df.copy().sort_values(["fuel", "date"])
    out["normal_distribution_margin_eur_l"] = (
        out.groupby("fuel", group_keys=False)["observed_distribution_margin_eur_l"]
        .transform(lambda s: s.rolling(window_weeks, min_periods=min_periods).median())
    )
    return out
