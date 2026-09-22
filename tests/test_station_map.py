import pandas as pd

from fuel_fair_price.visualization.station_map import build_station_map, local_excess_color


def test_local_excess_color_scale():
    assert local_excess_color(-8.0) == "#1a9850"
    assert local_excess_color(-2.0) == "#91cf60"
    assert local_excess_color(2.0) == "#ffffbf"
    assert local_excess_color(7.0) == "#fdae61"
    assert local_excess_color(15.0) == "#f46d43"
    assert local_excess_color(25.0) == "#d73027"
    assert local_excess_color(None) == "#9e9e9e"


def test_station_map_writes_interactive_html(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "id": "A",
                "fuel": "GAZOLE",
                "ville": "Paris",
                "adresse": "Test",
                "latitude_deg": 48.85,
                "longitude_deg": 2.35,
                "price_eur_l": 2.20,
                "local_fair_price_eur_l": 2.10,
                "local_excess_cent_l": 10.0,
                "local_anomaly_score": 3.0,
                "peer_confidence": "HIGH",
                "price_age_days": 1.0,
                "local_flag": "HIGH",
            },
            {
                "id": "B",
                "fuel": "SP95",
                "ville": "Lyon",
                "adresse": "Test",
                "latitude_deg": 45.76,
                "longitude_deg": 4.84,
                "price_eur_l": 1.90,
                "local_fair_price_eur_l": 1.95,
                "local_excess_cent_l": -5.0,
                "local_anomaly_score": -1.0,
                "peer_confidence": "MEDIUM",
                "price_age_days": 2.0,
                "local_flag": "NORMAL",
            },
        ]
    )
    path = build_station_map(frame, tmp_path / "station_map.html")
    text = path.read_text(encoding="utf-8")
    assert "Local price excess" in text
    assert "GAZOLE" in text and "assessed" in text
    assert "SP95" in text
    assert "Price anomaly indicator" in text
