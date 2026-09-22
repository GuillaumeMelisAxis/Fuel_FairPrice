import numpy as np
import pandas as pd

from fuel_fair_price.models.historical_local import (
    fit_historical_local_model,
    predict_structural_premium,
    validate_factor_ablation,
    validate_regime_splits,
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



def test_insufficient_logistics_is_diagnostic_only():
    panel = _synthetic_panel()
    panel.loc[panel["station_id"].isin(["0", "1"]), "accessibility_class"] = "FERRY_ISLAND"
    panel.loc[panel["station_id"].isin(["0", "1"]), "relative_price_cent_l"] += 20.0
    model, _, _ = fit_historical_local_model(panel)
    ferry = model[
        (model["fuel"] == "SP95")
        & (model["factor"] == "accessibility_class")
        & (model["level"] == "FERRY_ISLAND")
    ].iloc[0]
    assert ferry["confidence"] == "INSUFFICIENT"
    assert float(ferry["effect_cent_l"]) == 0.0
    assert float(ferry["provisional_effect_cent_l"]) >= 0.0
    assert ferry["status"] == "PROVISIONAL"


def test_sp95_road_uses_route_reference_and_no_motorway_discount():
    panel = _synthetic_panel()
    # Make a sparse motorway sample slightly cheaper than ROUTE: v0.5.1 must
    # not turn that into an economic motorway discount.
    auto_ids = ["0", "1", "2", "3"]
    panel.loc[panel["station_id"].isin(auto_ids), "road_type"] = "AUTOROUTE"
    panel.loc[panel["station_id"].isin(auto_ids), "relative_price_cent_l"] -= 2.0
    model, _, _ = fit_historical_local_model(panel)
    road = model[(model["fuel"] == "SP95") & (model["factor"] == "road_type")]
    route = road[road["level"] == "ROUTE"].iloc[0]
    auto = road[road["level"] == "AUTOROUTE"].iloc[0]
    assert float(route["effect_cent_l"]) == 0.0
    assert float(auto["effect_cent_l"]) >= 0.0
    assert "REFERENCE" in route["constraint_applied"]
    assert "NONNEGATIVE" in auto["constraint_applied"]


def test_persistent_bias_status_requires_twelve_months():
    rows = []
    months = pd.date_range("2025-01-31", periods=14, freq="ME")
    for date in months:
        for i in range(20):
            rows.append({
                "station_id": str(i), "date": date, "fuel": "SP95",
                "relative_price_cent_l": 12.0 if i == 0 else 0.0,
                "relative_price_robust_score": 1.0,
                "code_region": "A", "code_departement": "D",
                "road_type": "ROUTE", "competition_bucket": "MEDIUM",
                "isolation_bucket": "LOCAL", "accessibility_class": "MAINLAND",
            })
    panel = pd.DataFrame(rows)
    _, _, persistent = fit_historical_local_model(panel)
    p0 = persistent[persistent["station_id"] == "0"].iloc[0]
    assert int(p0["persistent_bias_months"]) == 14
    assert p0["persistence_status"] in {"ELEVATED", "HIGH", "VERY_HIGH"}
    assert p0["persistent_bias_confidence"] == "MEDIUM"


def test_regime_validation_and_ablation_return_expected_sections():
    rows = []
    months = pd.date_range("2025-01-31", periods=20, freq="ME")
    for date in months:
        for i in range(40):
            region = "A" if i < 20 else "B"
            rows.append({
                "station_id": str(i), "date": date, "fuel": "SP95",
                "relative_price_cent_l": -3.0 if region == "A" else 3.0,
                "relative_price_robust_score": 1.0,
                "code_region": region, "code_departement": "D1" if region == "A" else "D2",
                "road_type": "ROUTE", "competition_bucket": "MEDIUM",
                "isolation_bucket": "LOCAL", "accessibility_class": "MAINLAND",
            })
    panel = pd.DataFrame(rows)
    regime = validate_regime_splits(panel, shock_start="2026-02-28")
    assert {"PRE_SHOCK", "SHOCK_ONSET", "CURRENT_REGIME"}.issubset(set(regime["split"]))
    ablation = validate_factor_ablation(panel, holdout_months=3)
    assert "baseline" in set(ablation["model"])
    assert ablation["step"].max() == 6


def test_final_production_policy_keeps_sp95_road_and_logistics_diagnostic_only():
    panel = _synthetic_panel()
    panel.loc[panel["station_id"].isin(["0", "1"]), "road_type"] = "AUTOROUTE"
    panel.loc[panel["station_id"].isin(["0", "1"]), "accessibility_class"] = "FERRY_ISLAND"
    panel.loc[panel["station_id"].isin(["0", "1"]), "relative_price_cent_l"] += 10.0

    model, meta, _ = fit_historical_local_model(panel)
    fmeta = meta["fuels"]["SP95"]
    assert "road_type" not in fmeta["active_factors"]
    assert "accessibility_class" not in fmeta["active_factors"]
    assert "road_type" in fmeta["diagnostic_factors"]
    assert "accessibility_class" in fmeta["diagnostic_factors"]

    diag = model[
        (model["fuel"] == "SP95")
        & (model["factor"].isin(["road_type", "accessibility_class"]))
    ]
    assert not diag.empty
    assert (~diag["is_active"]).all()
    assert set(diag["role"]) == {"DIAGNOSTIC_ONLY"}

    sample = panel.iloc[[2, 3]].copy()
    # Same active characteristics, deliberately different diagnostic characteristics.
    sample.iloc[0, sample.columns.get_loc("road_type")] = "ROUTE"
    sample.iloc[0, sample.columns.get_loc("accessibility_class")] = "MAINLAND"
    sample.iloc[1, sample.columns.get_loc("road_type")] = "AUTOROUTE"
    sample.iloc[1, sample.columns.get_loc("accessibility_class")] = "FERRY_ISLAND"
    pred = predict_structural_premium(sample, fuel="SP95", model=model, meta=meta)
    assert np.isclose(
        pred.iloc[0]["structural_local_premium_cent_l"],
        pred.iloc[1]["structural_local_premium_cent_l"],
    )
    assert (pred["logistics_applied_to_fair_price"] == False).all()  # noqa: E712
    assert (pred["road_type_applied_to_fair_price"] == False).all()  # noqa: E712


def test_final_production_policy_keeps_gazole_road_active():
    panel = _synthetic_panel().copy()
    panel["fuel"] = "GAZOLE"
    panel.loc[panel["station_id"].isin(["0", "1", "2", "3", "4"]), "road_type"] = "AUTOROUTE"
    panel.loc[panel["station_id"].isin(["0", "1", "2", "3", "4"]), "relative_price_cent_l"] += 8.0

    model, meta, _ = fit_historical_local_model(panel)
    fmeta = meta["fuels"]["GAZOLE"]
    assert "road_type" in fmeta["active_factors"]
    assert "accessibility_class" not in fmeta["active_factors"]
    road = model[(model["fuel"] == "GAZOLE") & (model["factor"] == "road_type")]
    assert road["is_active"].all()
