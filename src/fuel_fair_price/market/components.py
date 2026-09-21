from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "fuel",
    "refined_quote_eur_l",
    "observed_distribution_margin_eur_l",
}


def _derive_effective_market_date(row: pd.Series) -> pd.Timestamp:
    """Best available date represented by a DGEC market row.

    For provisional rows such as ``provisional_to_2026-09-11`` the explicit
    cutoff is the correct live anchor date.  Older finalized monthly rows keep
    their nominal date because they are mainly used for historical analysis,
    not for today's live bridge.
    """
    status = str(row.get("source_status", ""))
    match = re.search(r"provisional_to_(\d{4}-\d{2}-\d{2})", status)
    if match:
        return pd.Timestamp(match.group(1))
    return pd.Timestamp(row["date"])


def load_market_components(path: str | Path) -> pd.DataFrame:
    """Load DGEC refined-product quotations and transport/distribution margins."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in market components: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["fuel"] = df["fuel"].str.upper()
    df["effective_market_date"] = df.apply(_derive_effective_market_date, axis=1)
    return df.sort_values(["fuel", "effective_market_date", "date"]).reset_index(drop=True)


def add_normal_distribution_margin(
    df: pd.DataFrame,
    window_periods: int = 12,
    min_periods: int = 3,
    baseline_by_fuel: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate a lagged robust normal transport/distribution margin.

    The current observation is excluded.  ``baseline_by_fuel`` is used until
    enough prior observations are available.
    """
    out = df.copy().sort_values(["fuel", "date"]).reset_index(drop=True)
    out["normal_distribution_margin_eur_l"] = float("nan")

    baseline_by_fuel = {k.upper(): v for k, v in (baseline_by_fuel or {}).items()}

    for fuel, idx in out.groupby("fuel", sort=False).groups.items():
        s = out.loc[idx, "observed_distribution_margin_eur_l"].astype(float)
        lagged = s.shift(1)
        normal = lagged.rolling(window_periods, min_periods=min_periods).median()
        baseline = baseline_by_fuel.get(str(fuel).upper())
        if baseline is not None:
            normal = normal.fillna(float(baseline))
        out.loc[idx, "normal_distribution_margin_eur_l"] = normal.to_numpy()

    return out
