from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuel_fair_price.data.historical_panel import build_monthly_station_panel
from fuel_fair_price.data.station_archive import download_annual_zip, parse_annual_zip
from fuel_fair_price.models.logistics import load_logistics_overrides

ROOT = Path(__file__).resolve().parents[1]


def _default_end_month() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None)
    return (now.to_period("M") - 1).to_timestamp("M")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v0.5 monthly station panel from official annual archives."
    )
    parser.add_argument("--start", default="2025-01", help="First month, YYYY-MM")
    parser.add_argument(
        "--end",
        default=str(_default_end_month().to_period("M")),
        help="Last complete month, YYYY-MM",
    )
    parser.add_argument(
        "--max-price-age-days",
        type=float,
        default=30.0,
        help="Drop carried prices older than this at each month-end.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "data" / "raw" / "annual"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "historical_station_panel.csv.gz"),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    start = pd.Period(args.start, freq="M").to_timestamp("M")
    end = pd.Period(args.end, freq="M").to_timestamp("M")
    if end < start:
        raise ValueError("--end must be >= --start")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # One carry year is downloaded so stations that did not change price in the
    # first training month still have a valid last-known price.
    years = list(range(start.year - 1, end.year + 1))
    event_frames = []

    for year in years:
        zip_path = cache_dir / f"prix_carburants_{year}.zip"
        if args.force_download or not zip_path.exists():
            print(f"Downloading official annual archive {year}...")
            download_annual_zip(year, zip_path)
        else:
            print(f"Using cached archive {zip_path}")

        print(f"Parsing {year}...")
        events = parse_annual_zip(zip_path)
        if not events.empty:
            event_frames.append(events)
            print(
                f"  {len(events):,} price-change events | "
                f"{events['station_id'].nunique():,} stations"
            )

    if not event_frames:
        raise RuntimeError("No annual price events were parsed")

    events = pd.concat(event_frames, ignore_index=True)
    # Limit the carry history to the last event before the start for each station/fuel,
    # plus all events inside the requested training window.
    before = events[events["updated_at"] < start].copy()
    carry = (
        before.sort_values("updated_at")
        .groupby(["station_id", "fuel"], as_index=False)
        .tail(1)
    )
    in_window = events[
        (events["updated_at"] >= start)
        & (events["updated_at"] <= end + pd.Timedelta(days=1))
    ].copy()
    events = pd.concat([carry, in_window], ignore_index=True)

    overrides = load_logistics_overrides(ROOT / "config" / "logistics_overrides.csv")

    from fuel_fair_price.data.historical_panel import HistoricalPanelConfig

    print("Building month-end network snapshots and competition features...")
    panel = build_monthly_station_panel(
        events,
        start=start,
        end=end,
        logistics_overrides=overrides,
        config=HistoricalPanelConfig(
            max_price_age_days=float(args.max_price_age_days)
        ),
    )
    if panel.empty:
        raise RuntimeError("Historical panel is empty")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False, compression="gzip")

    summary = (
        panel.groupby(["date", "fuel"], as_index=False)
        .agg(
            stations=("station_id", "nunique"),
            median_price_eur_l=("price_eur_l", "median"),
            median_price_age_days=("price_age_days", "median"),
        )
    )
    summary_path = ROOT / "output" / "historical_panel_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print()
    print(f"Saved panel: {output}")
    print(f"Rows: {len(panel):,}")
    print(f"Months: {panel['date'].nunique()}")
    print(panel.groupby("fuel")["station_id"].nunique())
    print()
    print(summary.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
