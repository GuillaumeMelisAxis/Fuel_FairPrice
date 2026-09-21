from pathlib import Path

from fuel_fair_price.market.eu_history import (
    load_france_weekly_htt,
)


ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_france_weekly_htt(
        start="2020-01-01",
    )

    output = (
        ROOT
        / "config"
        / "eu_france_weekly_htt.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved {len(df)} rows to {output}"
    )
    print()
    print(df.groupby("fuel").size())
    print()
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
