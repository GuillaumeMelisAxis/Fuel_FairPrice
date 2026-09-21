from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuel_fair_price.market.components import load_market_components
from fuel_fair_price.market.daily_proxy import (
    build_anchored_proxy_nowcast,
    extend_nowcast_daily,
    fit_brent_bridge_betas,
    latest_daily_market_snapshot,
    load_daily_product_proxies,
)


def load_baseline_margins(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path)
    latest = df.sort_values("year").groupby("fuel", as_index=False).tail(1)
    return dict(
        zip(
            latest["fuel"].str.upper(),
            latest["distribution_margin_eur_l"].astype(float),
        )
    )


def load_market_model(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fuel", "selected_filter"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing market model columns: {sorted(missing)}")
    df["fuel"] = df["fuel"].str.upper()
    return df


def build_live_market_snapshot(
    project_root: str | Path,
    *,
    target_date: str | pd.Timestamp | None = None,
    bridge_beta_start: str = "2023-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the daily refined-price nowcast and one live row per fuel."""
    root = Path(project_root)
    target = (
        pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()
        if target_date is None
        else pd.Timestamp(target_date).normalize()
    )

    anchors = load_market_components(root / "config" / "market_components.csv")
    model = load_market_model(root / "config" / "market_model.csv")
    selected = dict(zip(model["fuel"], model["selected_filter"]))

    latest_anchor = anchors["effective_market_date"].max()
    start = (min(latest_anchor, target) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    daily_proxy = load_daily_product_proxies(start=start)
    proxy_nowcast = build_anchored_proxy_nowcast(daily_proxy, anchors, selected)

    if "bridge_beta" in model.columns and model["bridge_beta"].notna().all():
        betas = dict(zip(model["fuel"], model["bridge_beta"].astype(float)))
    else:
        betas = fit_brent_bridge_betas(start=bridge_beta_start)

    extended = extend_nowcast_daily(
        proxy_nowcast,
        target_date=target,
        bridge_betas=betas,
    )
    snapshot = latest_daily_market_snapshot(extended, target_date=target)
    return extended, snapshot
