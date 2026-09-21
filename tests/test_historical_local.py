import numpy as np
import pandas as pd

from fuel_fair_price.models.historical_local import (
    fit_historical_local_model,
    predict_structural_premium,
)


def _synthetic_panel():
    rows = []
    months = pd.date_range("2025-01-31", periods=8, freq="ME")
    # 40 stations, half in each region. Region B is persistently +8 c/L.
    for date in months:
        for i in range(40):
            region = "A" if i < 20 else "B"
            relative = (-4.0 if region == "A" else 4.0)
            rows.append(
                {
                    "station_id": str(i),
                    "date": date,
                    "fuel": "SP95",
                    "relative_price_cent_l": relative,
                    "relative_price_robust_score": relative / 2.0,
                    "code_region": region,
                    "code_departement": "D1" if region == "A" else "D2",
                    "road_type": "ROUTE",
                    "competition_bucket": "MEDIUM",
                    "isolation_bucket": "LOCAL",
                    "accessibility_class": "MAINLAND",
                }
            )
    return pd.DataFrame(rows)


def test_historical_model_learns_persistent_regional_structure():
    panel = _synthetic_panel()
    model, meta, persistent = fit_historical_local_model(panel)

    b = model[
        (model["fuel"] == "SP95")
        & (model["factor"] == "code_region")
        & (model["level"] == "B")
    ]
    assert not b.empty
    assert float(b.iloc[0]["effect_cent_l"]) > 0.0
    assert meta["station_fixed_effect_applied_to_fair_price"] is False
    assert not persistent.empty


def test_accessibility_confidence_uses_unique_stations_not_observations():
    panel = _synthetic_panel()
    # Reclassify only two stations as ferry islands over all 8 months.
    panel.loc[panel["station_id"].isin(["0", "1"]), "accessibility_class"] = "FERRY_ISLAND"
    panel.loc[panel["station_id"].isin(["0", "1"]), "relative_price_cent_l"] += 12.0

    model, _, _ = fit_historical_local_model(panel)
    ferry = model[
        (model["factor"] == "accessibility_class")
        & (model["level"] == "FERRY_ISLAND")
    ].iloc[0]

    assert int(ferry["n_stations"]) == 2
    assert int(ferry["n_obs"]) == 16
    assert ferry["confidence"] == "INSUFFICIENT"
    assert float(ferry["effect_cent_l"]) >= 0.0


def test_prediction_is_centered_and_does_not_use_persistent_bias():
    panel = _synthetic_panel()
    model, meta, persistent = fit_historical_local_model(panel)
    sample = panel.iloc[:10].copy()
    pred = predict_structural_premium(
        sample,
        fuel="SP95",
        model=model,
        meta=meta,
    )
    assert "structural_local_premium_cent_l" in pred.columns
    assert "persistent_station_bias_cent_l" not in pred.columns
    assert np.isfinite(pred["structural_local_premium_cent_l"]).all()
