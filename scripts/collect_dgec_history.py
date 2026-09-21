from __future__ import annotations

import argparse
from pathlib import Path

from fuel_fair_price.market.dgec import collect_finalized_months


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect finalized DGEC NPG monthly market components")
    parser.add_argument("--start", required=True, help="First month, e.g. 2025-01")
    parser.add_argument("--end", required=True, help="Last month, e.g. 2026-08")
    parser.add_argument("--output", default="config/market_components.csv")
    parser.add_argument("--cache", default="data/cache/npg")
    args = parser.parse_args()

    df = collect_finalized_months(args.start, args.end, cache_dir=args.cache)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
