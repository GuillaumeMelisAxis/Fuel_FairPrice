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
    """Load DGEC refined-product quotations and transport/distribution margins."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in market components: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["fuel"] = df["fuel"].str.upper()
    return df.sort_values(["fuel", "date"]).reset_index(drop=True)


def add_normal_distribution_margin(
    df: pd.DataFrame,
    window_periods: int = 12,
    min_periods: int = 3,
    baseline_by_fuel: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Estimate a lagged robust 'normal' transport/distribution margin.

    The current observation is deliberately excluded.  At date t, the fair-price
    benchmark may only use margins observed strictly before t.  This prevents a
    contemporaneous margin spike from mechanically redefining itself as normal.

    ``baseline_by_fuel`` is used until ``min_periods`` prior observations are
    available.  It is useful for bootstrapping from the previous year's official
    DGEC average margin.
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
