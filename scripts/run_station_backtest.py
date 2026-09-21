from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuel_fair_price.analytics.backtest import build_monthly_backtest, load_baseline_margins
from fuel_fair_price.analytics.stations import score_historical_station_snapshot
from fuel_fair_price.data.station_archive import download_annual_zip, month_end_station_snapshot, parse_annual_zip
from fuel_fair_price.market.components import load_market_components

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Station-level month-end historical fair-price backtest")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache" / "stations"))
    args = parser.parse_args()

    cache = Path(args.cache)
    frames = []
    for year in [args.year - 1, args.year]:
        zpath = cache / f"PrixCarburants_annuel_{year}.zip"
        if not zpath.exists():
            download_annual_zip(year, zpath)
        frames.append(parse_annual_zip(zpath))
    events = pd.concat(frames, ignore_index=True)
    snapshot = month_end_station_snapshot(events, args.year)

    market = load_market_components(ROOT / "config" / "market_components.csv")
    prices = pd.read_csv(ROOT / "data" / "bootstrap_national_prices_2026.csv", parse_dates=["date"])
    macro = pd.read_csv(ROOT / "data" / "bootstrap_macro_2026.csv", parse_dates=["date"])
    baseline = load_baseline_margins(str(ROOT / "config" / "baseline_margins.csv"))
    fair = build_monthly_backtest(market, prices, macro, baseline)

    scored = score_historical_station_snapshot(snapshot, fair)
    out = ROOT / "output" / f"station_anomalies_{args.year}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"Wrote {len(scored):,} station-month-fuel rows to {out}")
    print(scored[["date", "station_id", "cp", "road_type", "fuel", "price_eur_l", "fair_price_eur_l", "spread_cent_l", "anomaly_z"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
