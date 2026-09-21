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
    assert anomaly["local_anomaly_score"] > 3.0
    assert anomaly["local_flag"] == "VERY_HIGH"
    assert not effects.empty


def test_structural_premium_is_centered():
    df, _ = apply_local_adjustment(
        _synthetic_stations(),
        config=LocalAdjustmentConfig(min_peer_count=4),
    )
    assert abs(float(df["structural_local_premium_cent_l"].median())) < 1e-9


def _zero_mad_peer_sample(n=24):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"P{i}",
                "fuel": "SP95",
                "latitude": 48.85 + (i % 6) * 0.002,
                "longitude": 2.35 + (i // 6) * 0.002,
                "code_region": "11",
                "code_departement": "75",
                "road_type": "ROUTE",
                "price_eur_l": 2.20,
                "fair_price_eur_l": 2.10,
                "spread_cent_l": 10.0,
                "local_residual_cent_l": 10.0,
            }
        )
    # A modest +6 c/L excess that used to explode when MAD == 0.
    rows[0]["price_eur_l"] = 2.26
    rows[0]["spread_cent_l"] = 16.0
    rows[0]["local_residual_cent_l"] = 16.0
    return pd.DataFrame(rows)


def test_peer_scale_floor_prevents_exploding_zscore():
    from fuel_fair_price.models.local_adjustment import add_local_peer_scores

    df = add_local_peer_scores(
        _zero_mad_peer_sample(),
        config=LocalAdjustmentConfig(
            min_peer_count=15,
            robust_scale_floor_cent_l=2.0,
        ),
    )
    anomaly = df[df["id"] == "P0"].iloc[0]
    assert anomaly["local_peer_mad_cent_l"] == 0.0
    assert anomaly["local_peer_scale_cent_l"] == 2.0
    assert anomaly["local_excess_cent_l"] == 6.0
    assert anomaly["local_anomaly_score"] == 3.0
    assert anomaly["local_anomaly_score"] < 10.0


def test_default_requires_fifteen_peers():
    assert LocalAdjustmentConfig().min_peer_count == 15


def test_peer_confidence_is_exposed():
    from fuel_fair_price.models.local_adjustment import add_local_peer_scores

    df = add_local_peer_scores(_zero_mad_peer_sample())
    assert set(df["peer_confidence"]).issubset({"HIGH", "MEDIUM", "LOW", "INSUFFICIENT"})
    assert (df["peer_confidence"] != "INSUFFICIENT").all()
    assert df["local_excess_cent_l"].notna().all()



def test_ferry_island_uses_logistics_peers_and_not_mainland():
    from fuel_fair_price.models.local_adjustment import add_local_peer_scores

    rows = []
    # Mainland cluster near the coast with low residuals.
    for i in range(20):
        rows.append(
            {
                "id": f"M{i}",
                "fuel": "GAZOLE",
                "latitude": 46.70 + i * 0.001,
                "longitude": -2.20 + i * 0.001,
                "local_residual_cent_l": 10.0,
                "accessibility_class": "MAINLAND",
                "logistics_peer_group": "MAINLAND",
            }
        )
    # Ferry islands have systematically higher residuals.
    for i in range(6):
        rows.append(
            {
                "id": f"F{i}",
                "fuel": "GAZOLE",
                "latitude": 46.72 + i * 0.002,
                "longitude": -2.35 + i * 0.002,
                "local_residual_cent_l": 25.0 + i,
                "accessibility_class": "FERRY_ISLAND",
                "logistics_peer_group": "TEST_ISLAND" if i < 3 else "OTHER_ISLAND",
            }
        )

    df = add_local_peer_scores(
        pd.DataFrame(rows),
        config=LocalAdjustmentConfig(min_peer_count=15, min_logistics_peer_count=5),
    )
    ferry = df[df["id"] == "F0"].iloc[0]
    assert ferry["peer_selection_method"] == "FERRY_CLASS"
    assert ferry["peer_confidence"] == "LOW"
    assert ferry["local_flag"] in {"NORMAL", "REVIEW_LOW_CONFIDENCE"}


def test_logistics_premium_is_non_negative_for_ferry_class():
    from fuel_fair_price.models.logistics import fit_logistics_premium

    df = pd.DataFrame(
        {
            "fuel": ["SP95"] * 8,
            "accessibility_class": ["MAINLAND"] * 5 + ["FERRY_ISLAND"] * 3,
            "structural_local_premium_cent_l": [0.0] * 8,
            "local_fair_price_eur_l": [2.0] * 8,
            "local_residual_cent_l": [10, 11, 9, 10, 10, 5, 6, 7],
            "fair_price_eur_l": [2.0] * 8,
            "price_eur_l": [2.10, 2.11, 2.09, 2.10, 2.10, 2.05, 2.06, 2.07],
        }
    )
    out, diag = fit_logistics_premium(df)
    ferry = out[out["accessibility_class"] == "FERRY_ISLAND"]
    assert (ferry["logistics_premium_cent_l"] >= 0).all()
    assert not diag.empty


def test_accessibility_override_classifies_ile_d_yeu():
    from fuel_fair_price.models.logistics import add_accessibility_features

    overrides = pd.DataFrame(
        [
            {
                "match_type": "ville",
                "match_value": "L'Île-d'Yeu",
                "accessibility_class": "FERRY_ISLAND",
                "logistics_peer_group": "ILE_YEU",
            }
        ]
    )
    df = pd.DataFrame(
        {
            "ville": ["L'Île-d'Yeu"],
            "nearest_station_km": [20.0],
            "stations_within_20km": [0],
        }
    )
    out = add_accessibility_features(df, overrides=overrides)
    assert out.iloc[0]["accessibility_class"] == "FERRY_ISLAND"
    assert out.iloc[0]["logistics_peer_group"] == "ILE_YEU"
