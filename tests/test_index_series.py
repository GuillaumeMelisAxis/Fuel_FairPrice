import numpy as np
import pandas as pd

from fuel_fair_price.analytics.index_series import (
    HistoricalIndexConfig,
    build_chained_observed_index,
    build_historical_fundamental_fair_series,
    build_historical_refined_quote_series,
    combine_index_series,
)


def test_chained_index_uses_matched_stations_not_composition():
    frame = pd.DataFrame(
        [
            {"date": "2026-01-31", "fuel": "SP95", "station_id": "A", "price_eur_l": 1.0},
            {"date": "2026-01-31", "fuel": "SP95", "station_id": "B", "price_eur_l": 2.0},
            {"date": "2026-02-28", "fuel": "SP95", "station_id": "A", "price_eur_l": 1.1},
            {"date": "2026-02-28", "fuel": "SP95", "station_id": "B", "price_eur_l": 2.2},
            # New expensive station must not distort the chained growth rate.
            {"date": "2026-02-28", "fuel": "SP95", "station_id": "C", "price_eur_l": 5.0},
        ]
    )
    out = build_chained_observed_index(
        frame,
        config=HistoricalIndexConfig(min_common_stations=2),
    )
    assert len(out) == 2
    assert np.isclose(out.iloc[1]["observed_price_index"], 110.0)
    assert out.iloc[1]["observed_index_method"] == "MATCHED_STATIONS"
    assert int(out.iloc[1]["matched_station_count"]) == 2


def test_refined_series_uses_official_month_and_proxy_backcast():
    dates = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-31", "2026-02-28"]),
            "fuel": ["SP95", "SP95", "SP95"],
        }
    )
    proxy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-01-31", "2026-02-28"]),
            "fuel": ["SP95"] * 4,
            "proxy_eur_l": [0.5, 0.6, 0.66, 0.72],
            "proxy_ma3_eur_l": [0.5, 0.6, 0.66, 0.72],
            "proxy_ma5_eur_l": [0.5, 0.6, 0.66, 0.72],
            "proxy_ma7_eur_l": [0.5, 0.6, 0.66, 0.72],
        }
    )
    anchors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "effective_market_date": pd.to_datetime(["2026-01-01"]),
            "fuel": ["SP95"],
            "refined_quote_eur_l": [0.6],
        }
    )
    out = build_historical_refined_quote_series(
        dates,
        daily_proxy=proxy,
        anchors=anchors,
        selected_filter={"SP95": "proxy_eur_l"},
    )
    dec = out[out["date"] == pd.Timestamp("2025-12-31")].iloc[0]
    jan = out[out["date"] == pd.Timestamp("2026-01-31")].iloc[0]
    feb = out[out["date"] == pd.Timestamp("2026-02-28")].iloc[0]
    assert dec["refined_quote_source"] == "DGEC_PROXY_BACKCAST"
    assert np.isclose(dec["refined_quote_est_eur_l"], 0.5)
    assert jan["refined_quote_source"] == "DGEC_MONTHLY_OFFICIAL"
    assert np.isclose(jan["refined_quote_est_eur_l"], 0.6)
    assert feb["refined_quote_source"] == "DGEC_ANCHORED_PROXY"
    assert np.isclose(feb["refined_quote_est_eur_l"], 0.72)


def test_fundamental_fair_uses_contemporaneous_tax_wedge():
    dates = pd.DataFrame({"date": pd.to_datetime(["2026-01-31"]), "fuel": ["SP95"]})
    weekly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-26"]),
            "fuel": ["SP95"],
            "htt_eur_l": [0.80],
            "ttc_eur_l": [1.65],
        }
    )
    refined = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-31"]),
            "fuel": ["SP95"],
            "refined_quote_est_eur_l": [0.50],
            "refined_quote_source": ["DGEC_MONTHLY_OFFICIAL"],
            "refined_anchor_date": pd.to_datetime(["2026-01-01"]),
            "refined_anchor_eur_l": [0.50],
            "selected_filter": ["proxy_eur_l"],
        }
    )
    out = build_historical_fundamental_fair_series(
        dates,
        weekly_france_prices=weekly,
        refined_series=refined,
        baseline_margins={"SP95": 0.25},
        config=HistoricalIndexConfig(vat_rate=0.20),
    )
    tax = 1.65 / 1.20 - 0.80
    expected = (0.50 + 0.25 + tax) * 1.20
    assert np.isclose(out.iloc[0]["effective_excise_eur_l"], tax)
    assert np.isclose(out.iloc[0]["fundamental_fair_price_eur_l"], expected)
    assert out.iloc[0]["fundamental_fair_confidence"] == "HIGH"


def test_combine_index_series_builds_market_tension_and_fair_index():
    observed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "fuel": ["SP95", "SP95"],
            "observed_median_price_eur_l": [2.0, 2.2],
            "observed_price_index": [100.0, 110.0],
        }
    )
    fair = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "fuel": ["SP95", "SP95"],
            "fundamental_fair_price_eur_l": [1.9, 2.0],
            "refined_quote_source": ["A", "A"],
            "fundamental_fair_confidence": ["HIGH", "HIGH"],
        }
    )
    anomaly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "fuel": ["SP95", "SP95"],
            "local_anomaly_rate_pct": [3.0, 4.0],
        }
    )
    out = combine_index_series(observed, fair, anomaly)
    assert np.isclose(out.iloc[0]["market_tension_cent_l"], 10.0)
    assert np.isclose(out.iloc[1]["market_tension_cent_l"], 20.0)
    assert np.isclose(out.iloc[0]["fundamental_fair_index"], 100.0)
    assert np.isclose(out.iloc[1]["fundamental_fair_index"], 100.0 * 2.0 / 1.9)


def test_live_market_snapshot_overrides_live_refined_quote_metadata():
    observed = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-22"]),
            "fuel": ["GAZOLE"],
            "observed_median_price_eur_l": [2.42],
            "observed_price_index": [140.0],
        }
    )
    fair = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-22"]),
            "fuel": ["GAZOLE"],
            "fundamental_fair_price_eur_l": [2.20],
            "refined_quote_est_eur_l": [1.06],
            "refined_quote_source": ["DGEC_ANCHORED_PROXY"],
            "fundamental_fair_confidence": ["MEDIUM"],
        }
    )
    anomaly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-22"]),
            "fuel": ["GAZOLE"],
            "local_anomaly_rate_pct": [5.0],
        }
    )
    live_summary = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-22"]),
            "fuel": ["GAZOLE"],
            "fair_price_eur_l": [2.34],
        }
    )
    live_market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-09-22"]),
            "fuel": ["GAZOLE"],
            "refined_nowcast_eur_l": [1.121],
        }
    )
    out = combine_index_series(
        observed,
        fair,
        anomaly,
        live_summary=live_summary,
        live_market_snapshot=live_market,
    )
    assert np.isclose(out.iloc[0]["fundamental_fair_price_eur_l"], 2.34)
    assert np.isclose(out.iloc[0]["refined_quote_est_eur_l"], 1.121)
    assert out.iloc[0]["refined_quote_source"] == "LIVE_V0_5_1"
