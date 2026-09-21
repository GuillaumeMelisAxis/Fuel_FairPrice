import json

import math
import pandas as pd

from fuel_fair_price.data.stations import clean_stations, to_long_format
from fuel_fair_price.market.components import load_market_components
from fuel_fair_price.market.freshness import assess_market_freshness


def test_raw_station_timestamp_is_paris_local():
    raw = pd.DataFrame(
        {
            "id": ["1"],
            "code_region": ["27"],
            "ville": ["Test"],
            "pop": ["R"],
            "prix": [json.dumps([
                {"@nom": "Gazole", "@maj": "2026-09-21 13:49:49", "@valeur": "2.419"},
                {"@nom": "SP95", "@maj": "2026-09-21 13:49:49", "@valeur": "2.199"},
            ])],
        }
    )
    out = to_long_format(clean_stations(raw))
    assert len(out) == 2
    assert str(out.iloc[0]["updated_at"].tzinfo) in {"Europe/Paris", "CEST"}
    assert set(out["fuel"]) == {"SP95", "GAZOLE"}


def test_market_freshness_current_bridge():
    f = assess_market_freshness(
        target_date="2026-09-21",
        effective_market_date="2026-09-21",
        product_proxy_date="2026-09-15",
        official_anchor_date="2026-09-11",
        market_mode="BRENT_FX_BRIDGE",
        bridge_source="YAHOO_DELAYED",
    )
    assert f.status == "CURRENT"
    assert f.bridge_span_days == 6
    assert f.official_anchor_age_days == 10


def test_market_freshness_very_stale():
    f = assess_market_freshness(
        target_date="2026-09-21",
        effective_market_date="2026-09-10",
        product_proxy_date="2026-09-10",
        official_anchor_date="2026-08-01",
        market_mode="BRENT_FX_BRIDGE",
        bridge_source="FRED_FALLBACK",
    )
    assert f.status == "VERY_STALE"


def test_market_component_effective_provisional_date(tmp_path):
    p = tmp_path / "market.csv"
    p.write_text(
        "date,fuel,refined_quote_eur_l,observed_distribution_margin_eur_l,source_status\n"
        "2026-09-01,SP95,0.8707,0.2380,provisional_to_2026-09-11\n"
    )
    df = load_market_components(p)
    assert df.iloc[0]["effective_market_date"] == pd.Timestamp("2026-09-11")


def test_daily_bridge_extends_product_proxy(monkeypatch):
    from fuel_fair_price.market import daily_proxy as dp

    proxy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-15"]),
            "fuel": ["GAZOLE"],
            "refined_nowcast_eur_l": [1.20],
            "official_anchor_date": pd.to_datetime(["2026-09-11"]),
            "official_anchor_eur_l": [1.06],
            "selected_filter": ["proxy_ma7_eur_l"],
            "market_mode": ["PRODUCT_PROXY"],
            "bridge_source": ["FRED_PRODUCT_PROXY"],
            "spot_observed_date": pd.to_datetime(["2026-09-15"]),
            "fx_observed_date": pd.to_datetime(["2026-09-15"]),
        }
    )

    bridge = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"]),
            "brent_usd_b": [100.0, 110.0, 120.0],
            "eurusd": [1.10, 1.10, 1.10],
            "brent_observed_date": pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"]),
            "fx_observed_date": pd.to_datetime(["2026-09-15", "2026-09-16", "2026-09-17"]),
            "bridge_source": ["TEST"] * 3,
        }
    )

    monkeypatch.setattr(dp, "load_live_bridge_market", lambda **kwargs: bridge)
    out = dp.extend_nowcast_daily(proxy, target_date="2026-09-17", bridge_betas={"GAZOLE": 1.0})
    latest = out.sort_values("date").iloc[-1]
    assert latest["market_mode"] == "BRENT_FX_BRIDGE"
    assert math.isclose(latest["refined_nowcast_eur_l"], 1.44, rel_tol=1e-12)
