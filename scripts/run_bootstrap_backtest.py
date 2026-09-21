from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuel_fair_price.analytics.backtest import backtest_summary, build_monthly_backtest, load_baseline_margins
from fuel_fair_price.market.components import load_market_components
from fuel_fair_price.visualization.charts import plot_margin, plot_market_components, plot_price_vs_fair

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    market = load_market_components(ROOT / "config" / "market_components.csv")
    prices = pd.read_csv(ROOT / "data" / "bootstrap_national_prices_2026.csv", parse_dates=["date"])
    macro = pd.read_csv(ROOT / "data" / "bootstrap_macro_2026.csv", parse_dates=["date"])
    baseline = load_baseline_margins(str(ROOT / "config" / "baseline_margins.csv"))

    bt = build_monthly_backtest(
        market=market,
        national_prices=prices,
        macro=macro,
        baseline_by_fuel=baseline,
    )

    outdir = ROOT / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(outdir / "bootstrap_backtest_2026.csv", index=False)
    summary = backtest_summary(bt)
    summary.to_csv(outdir / "bootstrap_backtest_summary_2026.csv", index=False)

    for fuel in ["SP95", "GAZOLE"]:
        plot_price_vs_fair(bt, fuel, outdir / f"{fuel.lower()}_price_vs_fair.png")
        plot_margin(bt, fuel, outdir / f"{fuel.lower()}_margin.png")
        plot_market_components(bt, fuel, outdir / f"{fuel.lower()}_market_components.png")

    cols = [
        "date", "fuel", "national_ttc_eur_l", "fair_price_eur_l",
        "price_residual_cent_l", "implied_distribution_margin_eur_l",
        "normal_distribution_margin_eur_l", "refined_quote_eur_l", "brent_eur_l",
    ]
    print(bt[cols].round(4).to_string(index=False))
    print("\nSummary")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
