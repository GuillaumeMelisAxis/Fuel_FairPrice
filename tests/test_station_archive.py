import pandas as pd

from fuel_fair_price.data.station_archive import month_end_station_snapshot


def test_month_end_station_snapshot_carries_last_price_for_complete_months():
    events = pd.DataFrame(
        {
            "station_id": ["1", "1", "1", "1"],
            "fuel": ["SP95", "SP95", "SP95", "SP95"],
            "updated_at": pd.to_datetime(["2025-12-20", "2026-01-15", "2026-03-10", "2026-04-05"]),
            "price_eur_l": [1.70, 1.75, 1.80, 1.82],
        }
    )
    snap = month_end_station_snapshot(events, 2026)
    jan = snap[snap["date"] == pd.Timestamp("2026-01-01")].iloc[0]
    feb = snap[snap["date"] == pd.Timestamp("2026-02-01")].iloc[0]
    mar = snap[snap["date"] == pd.Timestamp("2026-03-01")].iloc[0]
    assert jan.price_eur_l == 1.75
    assert feb.price_eur_l == 1.75
    assert mar.price_eur_l == 1.80
