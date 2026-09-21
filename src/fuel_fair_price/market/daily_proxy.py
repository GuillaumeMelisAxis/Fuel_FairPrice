from __future__ import annotations

from io import StringIO
from urllib.parse import quote
import time

import numpy as np
import pandas as pd
import requests

from fuel_fair_price.market.freshness import assess_market_freshness


FRED_SERIES = {
    "SP95": "DGASNYH",
    "GAZOLE": "DDFUELNYH",
    "EURUSD": "DEXUSEU",
    "BRENT": "DCOILBRENTEU",
}

YAHOO_SERIES = {
    "BRENT": "BZ=F",
    "EURUSD": "EURUSD=X",
}

LITRES_PER_US_GALLON = 3.785411784
PARIS_TZ = "Europe/Paris"


def fetch_fred_series(
    series_id: str,
    *,
    start: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")

    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]

    return df.reset_index(drop=True)


def load_daily_product_proxies(*, start: str = "2020-01-01") -> pd.DataFrame:
    """Public high-frequency refined-product proxies in EUR/L.

    They are used for dynamics only; the level is re-anchored to the official
    DGEC refined quotation.
    """
    fx = fetch_fred_series(FRED_SERIES["EURUSD"], start=start).rename(
        columns={"value": "eurusd"}
    )
    outputs = []

    for fuel in ("SP95", "GAZOLE"):
        spot = fetch_fred_series(FRED_SERIES[fuel], start=start).rename(
            columns={"value": "spot_usd_gal"}
        )

        first = max(spot["date"].min(), fx["date"].min())
        last = max(spot["date"].max(), fx["date"].max())
        calendar = pd.DataFrame({"date": pd.date_range(first, last, freq="D")})

        tmp = calendar.merge(spot, on="date", how="left").merge(fx, on="date", how="left")
        tmp["spot_observed_date"] = tmp["date"].where(tmp["spot_usd_gal"].notna())
        tmp["fx_observed_date"] = tmp["date"].where(tmp["eurusd"].notna())

        for col in ("spot_usd_gal", "eurusd", "spot_observed_date", "fx_observed_date"):
            tmp[col] = tmp[col].ffill(limit=4)

        tmp = tmp.dropna(subset=["spot_usd_gal", "eurusd"]).copy()
        tmp["proxy_eur_l"] = (
            tmp["spot_usd_gal"] / tmp["eurusd"] / LITRES_PER_US_GALLON
        )
        tmp["fuel"] = fuel

        for window in (3, 5, 7):
            tmp[f"proxy_ma{window}_eur_l"] = (
                tmp["proxy_eur_l"].rolling(window, min_periods=1).mean()
            )

        outputs.append(tmp)

    return pd.concat(outputs, ignore_index=True).sort_values(["fuel", "date"]).reset_index(drop=True)


def fetch_yahoo_daily(
    ticker: str,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch daily Yahoo chart data without yfinance.

    Yahoo is used only for the short live bridge.  If it is unavailable, the
    caller falls back to FRED and the freshness status will expose the staleness.
    """
    start_ts = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    period1 = int(start_ts.timestamp())
    period2 = int((end_ts + pd.Timedelta(days=1)).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includePrePost": "false",
    }
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 fuel-fair-price/0.3"},
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"No Yahoo chart result for {ticker}")

    node = result[0]
    timestamps = node.get("timestamp") or []
    quotes = (node.get("indicators", {}).get("quote") or [{}])[0]
    closes = quotes.get("close") or []
    if not timestamps or not closes:
        raise RuntimeError(f"No Yahoo daily closes for {ticker}")

    df = pd.DataFrame({"timestamp": timestamps, "value": closes})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(PARIS_TZ)
    df["date"] = dt.dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=["value"]).groupby("date", as_index=False)["value"].last().sort_values("date")


def _load_fred_bridge(*, start: str, end: pd.Timestamp) -> pd.DataFrame:
    brent = fetch_fred_series(FRED_SERIES["BRENT"], start=start).rename(
        columns={"value": "brent_usd_b"}
    )
    fx = fetch_fred_series(FRED_SERIES["EURUSD"], start=start).rename(
        columns={"value": "eurusd"}
    )
    return _combine_bridge_series(brent, fx, end=end, source="FRED_FALLBACK")


def _combine_bridge_series(
    brent: pd.DataFrame,
    fx: pd.DataFrame,
    *,
    end: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    first = min(brent["date"].min(), fx["date"].min())
    calendar = pd.DataFrame({"date": pd.date_range(first, end, freq="D")})
    out = calendar.merge(brent, on="date", how="left").merge(fx, on="date", how="left")
    out["brent_observed_date"] = out["date"].where(out["brent_usd_b"].notna())
    out["fx_observed_date"] = out["date"].where(out["eurusd"].notna())
    for col in ("brent_usd_b", "eurusd", "brent_observed_date", "fx_observed_date"):
        out[col] = out[col].ffill()
    out = out.dropna(subset=["brent_usd_b", "eurusd"]).copy()
    out["bridge_source"] = source
    return out


def load_live_bridge_market(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Latest Brent + EUR/USD bridge, Yahoo first and FRED as safe fallback."""
    end_ts = pd.Timestamp(end).normalize()
    start_ts = pd.Timestamp(start).normalize()

    try:
        brent = fetch_yahoo_daily(YAHOO_SERIES["BRENT"], start=start_ts, end=end_ts).rename(
            columns={"value": "brent_usd_b"}
        )
        fx = fetch_yahoo_daily(YAHOO_SERIES["EURUSD"], start=start_ts, end=end_ts).rename(
            columns={"value": "eurusd"}
        )
        return _combine_bridge_series(brent, fx, end=end_ts, source="YAHOO_DELAYED")
    except Exception:
        return _load_fred_bridge(start=start_ts.strftime("%Y-%m-%d"), end=end_ts)


def fit_brent_bridge_betas(
    *,
    start: str = "2023-01-01",
    min_observations: int = 200,
) -> dict[str, float]:
    """Calibrate product return sensitivity to Brent spot returns.

    The bridge models only *changes* after the last product-proxy date.  Betas
    are therefore fitted in USD return space and clipped to a conservative range.
    """
    brent = fetch_fred_series(FRED_SERIES["BRENT"], start=start).rename(
        columns={"value": "brent_usd_b"}
    )
    betas: dict[str, float] = {}

    for fuel in ("SP95", "GAZOLE"):
        product = fetch_fred_series(FRED_SERIES[fuel], start=start).rename(
            columns={"value": "spot_usd_gal"}
        )
        merged = product.merge(brent, on="date", how="inner").sort_values("date")
        merged["rp"] = np.log(merged["spot_usd_gal"]).diff()
        merged["rb"] = np.log(merged["brent_usd_b"]).diff()
        valid = merged[["rp", "rb"]].dropna()
        valid = valid[(valid["rp"].abs() < 0.20) & (valid["rb"].abs() < 0.20)]

        if len(valid) < min_observations:
            betas[fuel] = 1.0
            continue

        x = valid["rb"].to_numpy()
        y = valid["rp"].to_numpy()
        denom = float(np.dot(x, x))
        beta = float(np.dot(x, y) / denom) if denom > 0 else 1.0
        betas[fuel] = float(np.clip(beta, 0.20, 2.00))

    return betas


def build_anchored_proxy_nowcast(
    daily_proxy: pd.DataFrame,
    anchors: pd.DataFrame,
    selected_filter: dict[str, str],
) -> pd.DataFrame:
    """Anchor daily product dynamics to the latest official DGEC level."""
    anchors = anchors.copy()
    anchors["fuel"] = anchors["fuel"].str.upper()
    anchor_date_col = "effective_market_date" if "effective_market_date" in anchors.columns else "date"
    anchors[anchor_date_col] = pd.to_datetime(anchors[anchor_date_col])
    outputs = []

    for fuel, group in daily_proxy.groupby("fuel"):
        col = selected_filter.get(fuel, "proxy_ma5_eur_l")
        fuel_anchors = anchors[anchors["fuel"] == fuel].dropna(subset=["refined_quote_eur_l"]).sort_values(anchor_date_col)
        if fuel_anchors.empty:
            raise RuntimeError(f"No DGEC refined quote anchor for {fuel}")

        anchor = fuel_anchors.iloc[-1]
        anchor_date = pd.Timestamp(anchor[anchor_date_col]).normalize()
        official_quote = float(anchor["refined_quote_eur_l"])
        market = group.sort_values("date").copy()
        anchor_rows = market[market["date"] <= anchor_date].dropna(subset=[col])
        if anchor_rows.empty:
            raise RuntimeError(f"No product proxy at/before DGEC anchor for {fuel}")
        proxy_anchor = float(anchor_rows.iloc[-1][col])

        market["refined_nowcast_eur_l"] = official_quote * market[col] / proxy_anchor
        market["official_anchor_date"] = anchor_date
        market["official_anchor_eur_l"] = official_quote
        market["selected_filter"] = col
        market["market_mode"] = "PRODUCT_PROXY"
        market["bridge_source"] = "FRED_PRODUCT_PROXY"
        outputs.append(market)

    return pd.concat(outputs, ignore_index=True).sort_values(["fuel", "date"]).reset_index(drop=True)


def extend_nowcast_daily(
    proxy_nowcast: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
    bridge_betas: dict[str, float],
) -> pd.DataFrame:
    """Extend the anchored product nowcast to every calendar day up to target.

    When the public product proxy is stale, the extension uses Brent-futures
    returns and EUR/USD changes from the last proxy date.  The level remains
    anchored to DGEC via the already-built product nowcast.
    """
    target = pd.Timestamp(target_date).normalize()
    outputs = []

    for fuel, group in proxy_nowcast.groupby("fuel"):
        group = group.sort_values("date").copy()
        group = group[group["date"] <= target]
        if group.empty:
            continue

        last_proxy = group.iloc[-1]
        proxy_date = pd.Timestamp(last_proxy["date"]).normalize()
        outputs.append(group)

        if proxy_date >= target:
            continue

        bridge = load_live_bridge_market(start=proxy_date - pd.Timedelta(days=5), end=target)
        base_candidates = bridge[bridge["date"] <= proxy_date]
        if base_candidates.empty:
            continue
        base = base_candidates.iloc[-1]
        gamma = float(bridge_betas.get(fuel, 1.0))

        future = bridge[(bridge["date"] > proxy_date) & (bridge["date"] <= target)].copy()
        if future.empty:
            continue

        base_brent = float(base["brent_usd_b"])
        base_fx = float(base["eurusd"])
        base_quote = float(last_proxy["refined_nowcast_eur_l"])

        future["refined_nowcast_eur_l"] = (
            base_quote
            * np.power(future["brent_usd_b"] / base_brent, gamma)
            * (base_fx / future["eurusd"])
        )
        future["fuel"] = fuel
        future["official_anchor_date"] = last_proxy["official_anchor_date"]
        future["official_anchor_eur_l"] = last_proxy["official_anchor_eur_l"]
        future["selected_filter"] = last_proxy["selected_filter"]
        future["market_mode"] = "BRENT_FX_BRIDGE"
        future["bridge_beta"] = gamma
        outputs.append(future)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True, sort=False).sort_values(["fuel", "date"]).reset_index(drop=True)


def latest_daily_market_snapshot(
    nowcast: pd.DataFrame,
    *,
    target_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return one auditable live market row per fuel with freshness metadata."""
    target = pd.Timestamp(target_date).normalize()
    rows = []

    for fuel, group in nowcast.groupby("fuel"):
        group = group[group["date"] <= target].sort_values("date")
        if group.empty:
            continue
        latest = group.iloc[-1].copy()

        proxy_rows = group[group["market_mode"] == "PRODUCT_PROXY"]
        product_proxy_date = pd.Timestamp(proxy_rows["date"].max())

        if latest["market_mode"] == "BRENT_FX_BRIDGE":
            brent_obs = pd.Timestamp(latest.get("brent_observed_date"))
            fx_obs = pd.Timestamp(latest.get("fx_observed_date"))
            effective = min(brent_obs, fx_obs)
        else:
            spot_obs = pd.Timestamp(latest.get("spot_observed_date", latest["date"]))
            fx_obs = pd.Timestamp(latest.get("fx_observed_date", latest["date"]))
            effective = min(spot_obs, fx_obs)

        freshness = assess_market_freshness(
            target_date=target,
            effective_market_date=effective,
            product_proxy_date=product_proxy_date,
            official_anchor_date=latest["official_anchor_date"],
            market_mode=str(latest["market_mode"]),
            bridge_source=str(latest.get("bridge_source", "UNKNOWN")),
        )

        row = latest.to_dict()
        row.update(freshness.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)
