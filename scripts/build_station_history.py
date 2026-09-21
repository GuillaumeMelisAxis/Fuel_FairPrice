from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fuel_fair_price.data.station_archive import (
    download_annual_zip,
    month_end_station_snapshot,
    monthly_station_benchmark,
    parse_annual_zip,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build month-end SP95/Gazole station benchmarks")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--cache", default="data/cache/stations")
    parser.add_argument("--output", default=None)
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
    benchmark = monthly_station_benchmark(snapshot)
    output = Path(args.output or f"data/station_benchmark_{args.year}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output, index=False)
    print(f"Wrote {len(benchmark):,} benchmark rows to {output}")
    print(benchmark.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
