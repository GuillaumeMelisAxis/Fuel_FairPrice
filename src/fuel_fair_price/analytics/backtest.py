from __future__ import annotations

import numpy as np
import pandas as pd

from fuel_fair_price.market.components import add_normal_distribution_margin
from fuel_fair_price.models.brent import brent_usd_bbl_to_eur_l


def load_baseline_margins(path: str) -> dict[str, float]:
    df = pd.read_csv(path)
    latest = df.sort_values("year").groupby("fuel", as_index=False).tail(1)
    return dict(zip(latest["fuel"].str.upper(), latest["distribution_margin_eur_l"].astype(float)))


def build_monthly_backtest(
    market: pd.DataFrame,
    national_prices: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    baseline_by_fuel: dict[str, float] | None = None,
    vat_rate: float = 0.20,
    margin_window: int = 12,
    min_margin_history: int = 3,
) -> pd.DataFrame:
    """Build an ex-ante monthly fair-price backtest.

    Historical tax wedge is inferred from official HTT/TTC national series:
        effective_excise = TTC/(1+VAT) - HTT

    This intentionally isolates distribution-margin anomalies without requiring a
    hand-maintained history of exceptional rebates.  For the live index, the
    statutory tax schedule remains the preferred source.
    """
    m = market.copy()
    p = national_prices.copy()
    for df in (m, p):
        df["date"] = pd.to_datetime(df["date"])
        df["fuel"] = df["fuel"].str.upper()

    m = add_normal_distribution_margin(
        m,
        window_periods=margin_window,
        min_periods=min_margin_history,
        baseline_by_fuel=baseline_by_fuel,
    )
    out = p.merge(
        m[
            [
                "date", "fuel", "refined_quote_eur_l",
                "observed_distribution_margin_eur_l",
                "normal_distribution_margin_eur_l",
            ]
        ],
        on=["date", "fuel"],
        how="inner",
    )

    out["effective_excise_eur_l"] = (
        out["national_ttc_eur_l"] / (1.0 + vat_rate) - out["france_htt_eur_l"]
    )
    out["fair_price_eur_l"] = (
        out["refined_quote_eur_l"]
        + out["normal_distribution_margin_eur_l"]
        + out["effective_excise_eur_l"]
    ) * (1.0 + vat_rate)
    out["price_residual_eur_l"] = out["national_ttc_eur_l"] - out["fair_price_eur_l"]
    out["price_residual_cent_l"] = 100.0 * out["price_residual_eur_l"]
    out["implied_distribution_margin_eur_l"] = (
        out["national_ttc_eur_l"] / (1.0 + vat_rate)
        - out["effective_excise_eur_l"]
        - out["refined_quote_eur_l"]
    )

    if macro is not None and not macro.empty:
        macro = macro.copy()
        macro["date"] = pd.to_datetime(macro["date"])
        if "brent_eur_l" not in macro and {"brent_usd_bbl", "eurusd_usd_per_eur"}.issubset(macro.columns):
            macro["brent_eur_l"] = [
                brent_usd_bbl_to_eur_l(b, fx)
                for b, fx in zip(macro["brent_usd_bbl"], macro["eurusd_usd_per_eur"])
            ]
        out = out.merge(macro, on="date", how="left")
        if "brent_eur_l" in out:
            out["refined_vs_brent_eur_l"] = out["refined_quote_eur_l"] - out["brent_eur_l"]

    return out.sort_values(["fuel", "date"]).reset_index(drop=True)


def backtest_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fuel, g in backtest.groupby("fuel"):
        residual = g["price_residual_eur_l"].dropna()
        rows.append(
            {
                "fuel": fuel,
                "n_months": len(residual),
                "mae_cent_l": 100.0 * residual.abs().mean(),
                "rmse_cent_l": 100.0 * float(np.sqrt(np.mean(np.square(residual)))),
                "median_residual_cent_l": 100.0 * residual.median(),
                "max_over_fair_cent_l": 100.0 * residual.max(),
                "min_under_fair_cent_l": 100.0 * residual.min(),
            }
        )
    return pd.DataFrame(rows)
