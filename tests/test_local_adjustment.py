import numpy as np
import pandas as pd

from fuel_fair_price.models.local_adjustment import (
    LocalAdjustmentConfig,
    add_competition_features,
    apply_local_adjustment,
)


def _synthetic_stations():
    rows = []
    # Dense road cluster around Paris-like coordinates.
    for i in range(12):
        rows.append(
            {
                "id": f"R{i}",
                "fuel": "GAZOLE",
                "latitude": str(int((48.85 + i * 0.002) * 100000)),
                "longitude": str(int((2.35 + i * 0.002) * 100000)),
                "code_region": "11",
                "code_departement": "75",
                "road_type": "ROUTE",
                "price_eur_l": 2.40 + i * 0.001,
                "fair_price_eur_l": 2.30,
                "spread_cent_l": 10.0 + i * 0.1,
                "market_freshness": "CURRENT",
            }
        )
    # Autoroute cluster with a structural premium.
    for i in range(8):
        rows.append(
            {
                "id": f"A{i}",
                "fuel": "GAZOLE",
                "latitude": str(int((48.70 + i * 0.003) * 100000)),
                "longitude": str(int((2.10 + i * 0.003) * 100000)),
                "code_region": "11",
                "code_departement": "91",
                "road_type": "AUTOROUTE",
                "price_eur_l": 2.50 + i * 0.001,
                "fair_price_eur_l": 2.30,
                "spread_cent_l": 20.0 + i * 0.1,
                "market_freshness": "CURRENT",
            }
        )
    # One anomalous road station in dense cluster.
    rows[3]["price_eur_l"] = 2.70
    rows[3]["spread_cent_l"] = 40.0
    return pd.DataFrame(rows)


def test_competition_features_are_populated():
    df = add_competition_features(_synthetic_stations())
    assert df["nearest_station_km"].notna().all()
    assert (df["stations_within_10km"] >= 1).all()


def test_local_adjustment_detects_dense_cluster_outlier():
    df, effects = apply_local_adjustment(
        _synthetic_stations(),
        config=LocalAdjustmentConfig(min_peer_count=4),
    )
    anomaly = df[df["id"] == "R3"].iloc[0]
    assert anomaly["local_peer_z"] > 3.0
    assert anomaly["local_flag"] == "VERY_HIGH"
    assert not effects.empty


def test_structural_premium_is_centered():
    df, _ = apply_local_adjustment(
        _synthetic_stations(),
        config=LocalAdjustmentConfig(min_peer_count=4),
    )
    assert abs(float(df["structural_local_premium_cent_l"].median())) < 1e-9
