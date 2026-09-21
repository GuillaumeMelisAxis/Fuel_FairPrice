from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


FRED_SERIES = {
    # Product-level daily spot proxies.
    "SP95": "DGASNYH",
    "GAZOLE": "DDFUELNYH",
    # FX: USD per EUR.
    "EURUSD": "DEXUSEU",
}

LITRES_PER_US_GALLON = 3.785411784


def fetch_fred_series(
    series_id: str,
    *,
    start: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    url = (
        "https://fred.stlouisfed.org/graph/"
        f"fredgraph.csv?id={series_id}"
    )

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["date", "value"])
        .sort_values("date")
    )

    if start is not None:
        df = df[
            df["date"] >= pd.Timestamp(start)
        ]

    return df.reset_index(drop=True)


def load_daily_product_proxies(
    *,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Build daily refined-product proxies in EUR/L.

    SP95 proxy:
        EIA/FRED Conventional Gasoline, New York Harbor.

    Gazole proxy:
        EIA/FRED Ultra-Low-Sulfur No.2 Diesel, New York Harbor.

    These are NOT the official European DGEC quotations; they are liquid,
    public high-frequency product proxies used only for short-term dynamics.
    """
    fx = fetch_fred_series(
        FRED_SERIES["EURUSD"],
        start=start,
    ).rename(columns={"value": "eurusd"})

    outputs = []

    for fuel in ("SP95", "GAZOLE"):
        spot = fetch_fred_series(
            FRED_SERIES[fuel],
            start=start,
        ).rename(columns={"value": "spot_usd_gal"})

        first = min(
            spot["date"].min(),
            fx["date"].min(),
        )
        last = max(
            spot["date"].max(),
            fx["date"].max(),
        )

        calendar = pd.DataFrame(
            {
                "date": pd.date_range(
                    first,
                    last,
                    freq="D",
                )
            }
        )

        tmp = calendar.merge(
            spot,
            on="date",
            how="left",
        ).merge(
            fx,
            on="date",
            how="left",
        )

        # Carry last business observation through weekends/short holidays.
        tmp[["spot_usd_gal", "eurusd"]] = (
            tmp[["spot_usd_gal", "eurusd"]]
            .ffill(limit=4)
        )

        tmp = tmp.dropna(
            subset=["spot_usd_gal", "eurusd"]
        ).copy()

        tmp["proxy_eur_l"] = (
            tmp["spot_usd_gal"]
            / tmp["eurusd"]
            / LITRES_PER_US_GALLON
        )

        tmp["fuel"] = fuel

        # Moving averages in calendar days because pump prices are observed
        # every day. Weekends inherit Friday's last market observation.
        for window in (3, 5, 7):
            tmp[f"proxy_ma{window}_eur_l"] = (
                tmp["proxy_eur_l"]
                .rolling(window, min_periods=1)
                .mean()
            )

        outputs.append(tmp)

    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["fuel", "date"])
        .reset_index(drop=True)
    )


def build_anchored_nowcast(
    daily_proxy: pd.DataFrame,
    anchors: pd.DataFrame,
    selected_filter: dict[str, str],
) -> pd.DataFrame:
    """
    Anchor a public product proxy to the latest official DGEC refined quote.

        Q_nowcast(t)
          = Q_DGEC(anchor)
            * proxy_filter(t) / proxy_filter(anchor)

    Required anchor columns:
        date
        fuel
        refined_quote_eur_l
    """
    anchors = anchors.copy()
    anchors["date"] = pd.to_datetime(anchors["date"])

    outputs = []

    for fuel, group in daily_proxy.groupby("fuel"):
        col = selected_filter.get(
            fuel,
            "proxy_ma5_eur_l",
        )

        fuel_anchors = (
            anchors[anchors["fuel"] == fuel]
            .dropna(subset=["refined_quote_eur_l"])
            .sort_values("date")
        )

        if fuel_anchors.empty:
            raise RuntimeError(
                f"No DGEC refined-quote anchor for {fuel}."
            )

        anchor = fuel_anchors.iloc[-1]
        anchor_date = pd.Timestamp(anchor["date"])
        official_quote = float(
            anchor["refined_quote_eur_l"]
        )

        market = group.sort_values("date").copy()

        anchor_rows = market[
            market["date"] <= anchor_date
        ].dropna(subset=[col])

        if anchor_rows.empty:
            raise RuntimeError(
                f"No proxy observation at/before anchor date for {fuel}."
            )

        proxy_anchor = float(
            anchor_rows.iloc[-1][col]
        )

        market["refined_nowcast_eur_l"] = (
            official_quote
            * market[col]
            / proxy_anchor
        )

        market["anchor_date"] = anchor_date
        market["official_anchor_eur_l"] = official_quote
        market["selected_filter"] = col

        outputs.append(market)

    return (
        pd.concat(outputs, ignore_index=True)
        .sort_values(["fuel", "date"])
        .reset_index(drop=True)
    )
