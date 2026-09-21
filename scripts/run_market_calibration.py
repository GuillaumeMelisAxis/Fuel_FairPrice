from pathlib import Path

import pandas as pd

from fuel_fair_price.market.daily_proxy import (
    build_anchored_nowcast,
    load_daily_product_proxies,
)
from fuel_fair_price.models.lag_selection import (
    best_filter_by_fuel,
    build_calibration_panel,
    score_filters,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_anchor_file() -> pd.DataFrame:
    """
    Load the repo's existing market_components.csv.

    Expected information:
      date
      fuel
      refined quote in EUR/L

    A few common column names are accepted.
    """
    path = ROOT / "config" / "market_components.csv"
    df = pd.read_csv(path)

    date_col = next(
        (
            c for c in (
                "date",
                "period",
                "as_of_date",
            )
            if c in df.columns
        ),
        None,
    )
    fuel_col = next(
        (
            c for c in (
                "fuel",
                "carburant",
            )
            if c in df.columns
        ),
        None,
    )
    quote_col = next(
        (
            c for c in (
                "refined_quote_eur_l",
                "refined_product_eur_l",
                "quote_eur_l",
                "refined_eur_l",
            )
            if c in df.columns
        ),
        None,
    )

    if not all((date_col, fuel_col, quote_col)):
        raise RuntimeError(
            "Could not identify date/fuel/refined quote columns in "
            f"{path}. Columns are: {df.columns.tolist()}"
        )

    out = df[
        [date_col, fuel_col, quote_col]
    ].copy()

    out.columns = [
        "date",
        "fuel",
        "refined_quote_eur_l",
    ]

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )
    out["fuel"] = (
        out["fuel"]
        .astype(str)
        .str.upper()
    )
    out["refined_quote_eur_l"] = pd.to_numeric(
        out["refined_quote_eur_l"],
        errors="coerce",
    )

    return out.dropna()


def main():
    weekly_path = (
        ROOT
        / "config"
        / "eu_france_weekly_htt.csv"
    )

    weekly = pd.read_csv(
        weekly_path,
        parse_dates=["date"],
    )

    start = (
        weekly["date"].min()
        - pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d")

    daily = load_daily_product_proxies(
        start=start,
    )

    panel = build_calibration_panel(
        weekly,
        daily,
    )

    scores = score_filters(
        panel,
        train_fraction=0.75,
        min_train=52,
    )

    print("\nFILTER SCORES")
    print(scores.to_string(index=False))

    best = best_filter_by_fuel(scores)

    print("\nBEST FILTERS")
    for fuel, candidate in best.items():
        print(
            f"{fuel}: {candidate}"
        )

    if not best:
        raise RuntimeError(
            "No filter could be selected. "
            "Check the EU history parsing."
        )

    anchors = _load_anchor_file()

    nowcast = build_anchored_nowcast(
        daily,
        anchors,
        best,
    )

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    scores.to_csv(
        output_dir / "market_filter_scores.csv",
        index=False,
    )

    nowcast.to_csv(
        output_dir / "daily_refined_nowcast.csv",
        index=False,
    )

    print("\nLATEST NOWCAST")
    for fuel in ("SP95", "GAZOLE"):
        latest = (
            nowcast[nowcast["fuel"] == fuel]
            .sort_values("date")
            .iloc[-1]
        )

        print(
            f"{fuel}: "
            f"{latest['date'].date()} -> "
            f"{latest['refined_nowcast_eur_l']:.4f} EUR/L "
            f"(anchor {latest['anchor_date'].date()} = "
            f"{latest['official_anchor_eur_l']:.4f}, "
            f"filter={latest['selected_filter']})"
        )


if __name__ == "__main__":
    main()
