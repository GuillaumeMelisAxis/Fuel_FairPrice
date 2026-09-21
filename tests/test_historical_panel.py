import pandas as pd

from fuel_fair_price.data.historical_panel import (
    HistoricalPanelConfig,
    build_monthly_station_panel,
)


def test_panel_removes_common_monthly_price_level():
    events = []
    # Four stations; all prices rise by 20c between Jan and Feb, but cross-section stays same.
    for sid, offset in [("1", -0.03), ("2", -0.01), ("3", 0.01), ("4", 0.03)]:
        events.append(
            {
                "station_id": sid,
                "fuel": "SP95",
                "updated_at": pd.Timestamp("2025-01-20"),
                "price_eur_l": 1.80 + offset,
                "cp": "75001",
                "road_type": "ROUTE",
                "latitude": 48.85 + int(sid) * 0.001,
                "longitude": 2.35,
                "ville": "Paris",
            }
        )
        events.append(
            {
                "station_id": sid,
                "fuel": "SP95",
                "updated_at": pd.Timestamp("2025-02-20"),
                "price_eur_l": 2.00 + offset,
                "cp": "75001",
                "road_type": "ROUTE",
                "latitude": 48.85 + int(sid) * 0.001,
                "longitude": 2.35,
                "ville": "Paris",
            }
        )

    panel = build_monthly_station_panel(
        pd.DataFrame(events),
        start="2025-01",
        end="2025-02",
        config=HistoricalPanelConfig(max_price_age_days=30),
        add_geographic_features=False,
    )

    jan = panel[panel["date"] == pd.Timestamp("2025-01-31")].sort_values("station_id")
    feb = panel[panel["date"] == pd.Timestamp("2025-02-28")].sort_values("station_id")

    assert jan["relative_price_cent_l"].round(8).tolist() == feb["relative_price_cent_l"].round(8).tolist()
    assert abs(float(jan["relative_price_cent_l"].median())) < 1e-12
    assert abs(float(feb["relative_price_cent_l"].median())) < 1e-12
