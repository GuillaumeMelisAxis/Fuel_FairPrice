from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuel_fair_price.models.local_adjustment import (
    LocalAdjustmentConfig,
    add_competition_features,
    add_local_peer_scores,
)
from fuel_fair_price.models.logistics import (
    LogisticsConfig,
    add_accessibility_features,
)

CONFIDENCE_RANK = {
    "INSUFFICIENT": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


@dataclass(frozen=True)
class HistoricalLocalModelConfig:
    trim_abs_robust_score: float = 4.0
    n_backfit_iterations: int = 10
    region_shrinkage_stations: float = 80.0
    department_shrinkage_stations: float = 40.0
    road_shrinkage_stations: float = 100.0
    competition_shrinkage_stations: float = 80.0
    isolation_shrinkage_stations: float = 50.0
    accessibility_shrinkage_stations: float = 10.0
    persistent_bias_shrinkage_months: float = 6.0
    persistent_status_min_months: int = 12
    persistent_high_confidence_months: int = 18
    persistent_medium_confidence_months: int = 12
    persistent_low_confidence_months: int = 6
    high_confidence_stations: int = 100
    medium_confidence_stations: int = 30
    low_confidence_stations: int = 10


FACTOR_SPECS = (
    ("code_region", "region_effect_cent_l", "region_shrinkage_stations"),
    ("code_departement", "department_effect_cent_l", "department_shrinkage_stations"),
    ("road_type", "road_effect_cent_l", "road_shrinkage_stations"),
    ("competition_bucket", "competition_effect_cent_l", "competition_shrinkage_stations"),
    ("isolation_bucket", "isolation_effect_cent_l", "isolation_shrinkage_stations"),
    ("accessibility_class", "logistics_premium_cent_l", "accessibility_shrinkage_stations"),
)


FACTOR_SPEC_BY_NAME = {spec[0]: spec for spec in FACTOR_SPECS}


# Final v0.5.1 production specification selected from the out-of-sample ablation.
# Diagnostic factors are still estimated/exported, but never enter the fair price.
ACTIVE_FACTORS_BY_FUEL: dict[str, tuple[str, ...]] = {
    "GAZOLE": (
        "code_region",
        "code_departement",
        "road_type",
        "competition_bucket",
        "isolation_bucket",
    ),
    "SP95": (
        "code_region",
        "code_departement",
        "competition_bucket",
        "isolation_bucket",
    ),
}

DIAGNOSTIC_FACTORS_BY_FUEL: dict[str, tuple[str, ...]] = {
    "GAZOLE": ("accessibility_class",),
    "SP95": ("road_type", "accessibility_class"),
}


def _confidence_from_unique_stations(n: int, config: HistoricalLocalModelConfig) -> str:
    if n >= config.high_confidence_stations:
        return "HIGH"
    if n >= config.medium_confidence_stations:
        return "MEDIUM"
    if n >= config.low_confidence_stations:
        return "LOW"
    return "INSUFFICIENT"


def _persistent_confidence(months: int, config: HistoricalLocalModelConfig) -> str:
    if months >= config.persistent_high_confidence_months:
        return "HIGH"
    if months >= config.persistent_medium_confidence_months:
        return "MEDIUM"
    if months >= config.persistent_low_confidence_months:
        return "LOW"
    return "INSUFFICIENT"


def _persistent_status(bias_cent_l: float, months: int, config: HistoricalLocalModelConfig) -> str:
    if months < config.persistent_status_min_months:
        return "PROVISIONAL"
    if bias_cent_l > 15.0:
        return "VERY_HIGH"
    if bias_cent_l > 10.0:
        return "HIGH"
    if bias_cent_l > 5.0:
        return "ELEVATED"
    return "NORMAL"


def _selected_factor_specs(factors: tuple[str, ...] | list[str] | None):
    if factors is None:
        return FACTOR_SPECS
    unknown = [name for name in factors if name not in FACTOR_SPEC_BY_NAME]
    if unknown:
        raise KeyError(f"Unknown historical factors: {unknown}")
    return tuple(FACTOR_SPEC_BY_NAME[name] for name in factors)


def _stabilize_effect_stats(
    stats: pd.DataFrame,
    *,
    factor: str,
    fuel: str,
    config: HistoricalLocalModelConfig,
) -> pd.DataFrame:
    """Apply v0.5.1 economic/reliability constraints to fitted factor effects."""
    out = stats.copy()
    out["provisional_effect_cent_l"] = out["effect_cent_l"].astype(float)
    out["constraint_applied"] = "NONE"

    # Logistics effects based on fewer than LOW-confidence unique stations are
    # diagnostic only; they must not alter the live fair price.
    if factor == "accessibility_class":
        insufficient = out["n_stations"].astype(int) < int(config.low_confidence_stations)
        out.loc[insufficient, "effect_cent_l"] = 0.0
        out.loc[insufficient, "constraint_applied"] = "INSUFFICIENT_ZEROED"

    # For SP95, ROUTE is the economic reference category. We never allow the
    # sparse AUTOROUTE estimate to imply a discount relative to ROUTE.
    if factor == "road_type" and str(fuel) == "SP95":
        levels = out[factor].astype(str)
        route_rows = out.loc[levels == "ROUTE", "effect_cent_l"]
        route_effect = float(route_rows.iloc[0]) if len(route_rows) else 0.0
        out["effect_cent_l"] = out["effect_cent_l"].astype(float) - route_effect
        out.loc[levels == "ROUTE", "effect_cent_l"] = 0.0
        out.loc[levels == "ROUTE", "constraint_applied"] = "SP95_ROUTE_REFERENCE_ZERO"
        auto = levels == "AUTOROUTE"
        if auto.any():
            out.loc[auto, "effect_cent_l"] = out.loc[auto, "effect_cent_l"].clip(lower=0.0)
            out.loc[auto, "constraint_applied"] = "SP95_AUTOROUTE_NONNEGATIVE_PREMIUM"

    return out


def _panel_group_effect(
    frame: pd.DataFrame,
    *,
    factor: str,
    residual_col: str,
    shrinkage_stations: float,
    constrained_nonnegative: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Robust group effect using station-level medians.

    Repeated monthly observations from the same station do not artificially increase
    the shrinkage weight: the effect is shrunk using the number of *unique stations*.
    """
    work = frame[["station_id", factor, residual_col]].copy()
    work[factor] = work[factor].fillna("UNKNOWN").astype(str)
    work = work.dropna(subset=[residual_col])

    station_level = (
        work.groupby([factor, "station_id"], as_index=False)[residual_col]
        .median()
        .rename(columns={residual_col: "station_median_residual"})
    )

    group_stats = (
        station_level.groupby(factor, as_index=False)
        .agg(
            raw_effect_cent_l=("station_median_residual", "median"),
            n_stations=("station_id", "nunique"),
        )
    )

    n_obs = (
        work.groupby(factor, as_index=False)
        .size()
        .rename(columns={"size": "n_obs"})
    )
    group_stats = group_stats.merge(n_obs, on=factor, how="left")
    group_stats["effect_cent_l"] = (
        group_stats["raw_effect_cent_l"]
        * group_stats["n_stations"]
        / (group_stats["n_stations"] + float(shrinkage_stations))
    )

    if factor == "accessibility_class":
        mainland = group_stats[factor] == "MAINLAND"
        group_stats.loc[mainland, ["raw_effect_cent_l", "effect_cent_l"]] = 0.0

    if constrained_nonnegative:
        constrained_levels = {"FERRY_ISLAND", "ROAD_CONNECTED_ISLAND", "REMOTE_ISOLATED"}
        mask = group_stats[factor].isin(constrained_levels)
        group_stats.loc[mask, "effect_cent_l"] = (
            group_stats.loc[mask, "effect_cent_l"].clip(lower=0.0)
        )

    mapping = dict(
        zip(
            group_stats[factor].astype(str),
            group_stats["effect_cent_l"].astype(float),
        )
    )
    return mapping, group_stats


def _map_effect(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.fillna("UNKNOWN").astype(str).map(mapping).fillna(0.0).astype(float)


def fit_historical_local_model(
    panel: pd.DataFrame,
    *,
    config: HistoricalLocalModelConfig = HistoricalLocalModelConfig(),
    factors: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit the historical structural local-premium model.

    With ``factors=None`` this is the final v0.5.1 production specification:
    active factors are fuel-specific and only those factors enter the fair price.
    Diagnostic-only factors are estimated conditionally on the active model and
    exported for auditability, but never affect the structural premium.

    Passing an explicit ``factors`` list keeps the generic behaviour used by
    validation/ablation: all supplied factors are active.
    """
    if panel.empty:
        raise ValueError("Historical panel is empty")

    production_mode = factors is None
    required = {
        "station_id",
        "date",
        "fuel",
        "relative_price_cent_l",
        "relative_price_robust_score",
    }

    if production_mode:
        factor_names = set()
        for fuel_name in panel["fuel"].dropna().astype(str).unique():
            factor_names.update(ACTIVE_FACTORS_BY_FUEL.get(fuel_name, ()))
            factor_names.update(DIAGNOSTIC_FACTORS_BY_FUEL.get(fuel_name, ()))
    else:
        factor_names = set(str(x) for x in factors)
        _selected_factor_specs(tuple(factor_names))  # validation of names

    required.update(factor_names)
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Missing historical-panel columns: {sorted(missing)}")

    panel = panel.copy()
    panel["station_id"] = panel["station_id"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"])
    rows: list[dict] = []
    persistent_frames: list[pd.DataFrame] = []
    fuel_meta: dict[str, dict] = {}

    for fuel, fuel_panel in panel.groupby("fuel", sort=True):
        fuel = str(fuel)
        if production_mode:
            active_names = ACTIVE_FACTORS_BY_FUEL.get(
                fuel,
                tuple(spec[0] for spec in FACTOR_SPECS if spec[0] != "accessibility_class"),
            )
            diagnostic_names = DIAGNOSTIC_FACTORS_BY_FUEL.get(fuel, ("accessibility_class",))
        else:
            active_names = tuple(str(x) for x in factors)
            diagnostic_names = ()

        active_specs = _selected_factor_specs(active_names)
        diagnostic_specs = _selected_factor_specs(diagnostic_names) if diagnostic_names else ()
        all_specs = tuple(dict.fromkeys([*active_specs, *diagnostic_specs]))

        work = fuel_panel.copy()
        for factor, _, _ in all_specs:
            work[factor] = work[factor].fillna("UNKNOWN").astype(str)

        train = work[
            work["relative_price_robust_score"].abs()
            <= float(config.trim_abs_robust_score)
        ].copy()
        if train.empty:
            train = work.copy()

        active_maps: dict[str, dict[str, float]] = {}
        active_series = {
            effect_name: pd.Series(0.0, index=train.index)
            for _, effect_name, _ in active_specs
        }
        active_stats: dict[str, pd.DataFrame] = {}

        # Backfit only factors that are allowed to enter the production fair price.
        for _ in range(config.n_backfit_iterations):
            for factor, effect_name, shrink_attr in active_specs:
                other = sum(
                    (values for name, values in active_series.items() if name != effect_name),
                    start=pd.Series(0.0, index=train.index),
                )
                residual = train["relative_price_cent_l"] - other
                temp = train[["station_id", factor]].copy()
                temp["_residual"] = residual

                mapping, stats = _panel_group_effect(
                    temp,
                    factor=factor,
                    residual_col="_residual",
                    shrinkage_stations=float(getattr(config, shrink_attr)),
                    constrained_nonnegative=(factor == "accessibility_class"),
                )
                stats = _stabilize_effect_stats(
                    stats,
                    factor=factor,
                    fuel=fuel,
                    config=config,
                )
                mapping = dict(
                    zip(stats[factor].astype(str), stats["effect_cent_l"].astype(float))
                )
                active_maps[effect_name] = mapping
                active_stats[effect_name] = stats
                active_series[effect_name] = _map_effect(train[factor], mapping)

        train_pred_raw = sum(
            active_series.values(),
            start=pd.Series(0.0, index=train.index),
        )
        centre = float(train_pred_raw.median()) if len(train_pred_raw) else 0.0
        train_pred = train_pred_raw - centre

        # Export active factors.
        for factor, effect_name, shrink_attr in active_specs:
            stats = active_stats[effect_name].copy()
            for rec in stats.to_dict("records"):
                n_stations = int(rec["n_stations"])
                confidence = _confidence_from_unique_stations(n_stations, config)
                rows.append(
                    {
                        "fuel": fuel,
                        "factor": factor,
                        "effect_column": effect_name,
                        "level": str(rec[factor]),
                        "effect_cent_l": float(rec["effect_cent_l"]),
                        "provisional_effect_cent_l": float(
                            rec.get("provisional_effect_cent_l", rec["effect_cent_l"])
                        ),
                        "raw_effect_cent_l": float(rec["raw_effect_cent_l"]),
                        "n_obs": int(rec["n_obs"]),
                        "n_stations": n_stations,
                        "confidence": confidence,
                        "status": "PROVISIONAL" if confidence == "INSUFFICIENT" else "ESTIMATED",
                        "constraint_applied": str(rec.get("constraint_applied", "NONE")),
                        "shrinkage_stations": float(getattr(config, shrink_attr)),
                        "is_active": True,
                        "role": "ACTIVE",
                    }
                )

        # Estimate diagnostic-only factors on the residual left by the active model.
        # They are deliberately not backfitted into active effects and never change
        # the centre or the production fair price.
        diagnostic_maps: dict[str, dict[str, float]] = {}
        post_active_residual = train["relative_price_cent_l"] - train_pred
        for factor, effect_name, shrink_attr in diagnostic_specs:
            temp = train[["station_id", factor]].copy()
            temp["_residual"] = post_active_residual
            _, stats = _panel_group_effect(
                temp,
                factor=factor,
                residual_col="_residual",
                shrinkage_stations=float(getattr(config, shrink_attr)),
                constrained_nonnegative=(factor == "accessibility_class"),
            )
            stats = _stabilize_effect_stats(
                stats,
                factor=factor,
                fuel=fuel,
                config=config,
            )
            diagnostic_maps[effect_name] = dict(
                zip(stats[factor].astype(str), stats["effect_cent_l"].astype(float))
            )
            for rec in stats.to_dict("records"):
                n_stations = int(rec["n_stations"])
                confidence = _confidence_from_unique_stations(n_stations, config)
                rows.append(
                    {
                        "fuel": fuel,
                        "factor": factor,
                        "effect_column": effect_name,
                        "level": str(rec[factor]),
                        "effect_cent_l": float(rec["effect_cent_l"]),
                        "provisional_effect_cent_l": float(
                            rec.get("provisional_effect_cent_l", rec["effect_cent_l"])
                        ),
                        "raw_effect_cent_l": float(rec["raw_effect_cent_l"]),
                        "n_obs": int(rec["n_obs"]),
                        "n_stations": n_stations,
                        "confidence": confidence,
                        "status": "PROVISIONAL" if confidence == "INSUFFICIENT" else "ESTIMATED",
                        "constraint_applied": str(rec.get("constraint_applied", "NONE")),
                        "shrinkage_stations": float(getattr(config, shrink_attr)),
                        "is_active": False,
                        "role": "DIAGNOSTIC_ONLY",
                    }
                )

        # Persistent station bias is computed after ACTIVE factors only.
        pred_all_raw = pd.Series(0.0, index=work.index)
        for factor, effect_name, _ in active_specs:
            pred_all_raw += _map_effect(work[factor], active_maps[effect_name])
        pred_all = pred_all_raw - centre
        work["_structural_prediction"] = pred_all
        work["_post_model_residual"] = (
            work["relative_price_cent_l"] - work["_structural_prediction"]
        )

        persistent = (
            work.groupby("station_id", as_index=False)
            .agg(
                persistent_station_bias_raw_cent_l=("_post_model_residual", "median"),
                persistent_bias_months=("date", "nunique"),
                persistent_bias_observations=("_post_model_residual", "count"),
            )
        )
        persistent["persistent_station_bias_cent_l"] = (
            persistent["persistent_station_bias_raw_cent_l"]
            * persistent["persistent_bias_months"]
            / (
                persistent["persistent_bias_months"]
                + float(config.persistent_bias_shrinkage_months)
            )
        )
        persistent["persistent_bias_confidence"] = persistent["persistent_bias_months"].map(
            lambda n: _persistent_confidence(int(n), config)
        )
        persistent["persistence_status"] = persistent.apply(
            lambda row: _persistent_status(
                float(row["persistent_station_bias_cent_l"]),
                int(row["persistent_bias_months"]),
                config,
            ),
            axis=1,
        )
        persistent["fuel"] = fuel

        last_meta_cols = [
            c for c in ["ville", "cp", "code_departement", "code_region"]
            if c in work.columns
        ]
        if last_meta_cols:
            latest = (
                work.sort_values("date")
                .groupby("station_id", as_index=False)
                .tail(1)[["station_id", *last_meta_cols]]
            )
            persistent = persistent.merge(latest, on="station_id", how="left")

        persistent_frames.append(persistent)
        fuel_meta[fuel] = {
            "centre_cent_l": centre,
            "train_start": str(pd.Timestamp(train["date"].min()).date()),
            "train_end": str(pd.Timestamp(train["date"].max()).date()),
            "n_observations": int(len(train)),
            "n_unique_stations": int(train["station_id"].nunique()),
            "n_months": int(train["date"].nunique()),
            "active_factors": list(active_names),
            "diagnostic_factors": list(diagnostic_names),
        }

    model = pd.DataFrame(rows)
    persistent_bias = (
        pd.concat(persistent_frames, ignore_index=True)
        if persistent_frames else pd.DataFrame()
    )
    meta = {
        "model_version": "0.5.1-final",
        "model_type": "HISTORICAL_PANEL",
        "target": "relative_price_cent_l",
        "time_effect": "date_x_fuel_national_station_median_removed",
        "station_fixed_effect_applied_to_fair_price": False,
        "production_factor_policy": "fuel_specific_active_factors_from_oos_ablation",
        "active_factors_by_fuel": {
            k: list(v) for k, v in ACTIVE_FACTORS_BY_FUEL.items()
        } if production_mode else None,
        "diagnostic_factors_by_fuel": {
            k: list(v) for k, v in DIAGNOSTIC_FACTORS_BY_FUEL.items()
        } if production_mode else None,
        "factors": list(factors) if factors is not None else [],
        "stabilization": {
            "accessibility_class_diagnostic_only": bool(production_mode),
            "sp95_road_type_diagnostic_only": bool(production_mode),
            "insufficient_logistics_applied_effect_zero": True,
            "sp95_road_reference": "ROUTE=0; AUTOROUTE premium constrained >= 0",
            "persistent_bias_applied_to_fair_price": False,
        },
        "config": asdict(config),
        "fuels": fuel_meta,
    }
    return model, meta, persistent_bias

def _model_maps_for_fuel(
    model: pd.DataFrame,
    fuel: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    part = model[model["fuel"].astype(str) == str(fuel)].copy()
    maps: dict[str, dict[str, float]] = {}
    confidence: dict[str, dict[str, str]] = {}
    for factor, group in part.groupby("factor"):
        maps[str(factor)] = dict(
            zip(group["level"].astype(str), group["effect_cent_l"].astype(float))
        )
        confidence[str(factor)] = dict(
            zip(group["level"].astype(str), group["confidence"].astype(str))
        )
    return maps, confidence


def predict_structural_premium(
    frame: pd.DataFrame,
    *,
    fuel: str,
    model: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    out = frame.copy()
    fuel = str(fuel)
    maps, confidence_maps = _model_maps_for_fuel(model, fuel)
    fuel_meta = meta["fuels"][fuel]
    centre = float(fuel_meta["centre_cent_l"])

    # Final production models carry fuel-specific active/diagnostic factor lists.
    # Explicit ablation models use the legacy global ``factors`` list.
    active_names = tuple(
        fuel_meta.get("active_factors")
        or meta.get("factors", [])
    )
    diagnostic_names = tuple(fuel_meta.get("diagnostic_factors", []))
    mapped_names = tuple(dict.fromkeys([*active_names, *diagnostic_names]))
    mapped_specs = _selected_factor_specs(mapped_names) if mapped_names else ()

    # Stable output schema: every known effect column exists.
    for _, effect_name, _ in FACTOR_SPECS:
        if effect_name not in out.columns:
            out[effect_name] = 0.0

    active_effect_cols: list[str] = []
    active_coverage_cols: list[str] = []
    active_confidence_cols: list[str] = []
    all_coverage_cols: list[str] = []

    for factor, effect_name, _ in mapped_specs:
        values = out.get(
            factor,
            pd.Series("UNKNOWN", index=out.index),
        ).fillna("UNKNOWN").astype(str)
        mapping = maps.get(factor, {})
        conf_mapping = confidence_maps.get(factor, {})

        out[effect_name] = values.map(mapping).fillna(0.0).astype(float)
        matched = values.isin(mapping.keys())
        coverage_col = f"{factor}_model_match"
        out[coverage_col] = matched
        all_coverage_cols.append(coverage_col)

        ccol = f"{factor}_model_confidence"
        out[ccol] = values.map(conf_mapping).fillna("INSUFFICIENT")

        if factor in active_names:
            active_effect_cols.append(effect_name)
            active_coverage_cols.append(coverage_col)
            active_confidence_cols.append(ccol)

    out["base_structural_local_premium_cent_l"] = (
        out[active_effect_cols].sum(axis=1) - centre
        if active_effect_cols
        else -centre
    )
    # Final v0.5.1: diagnostics never enter the fair price.
    out["structural_local_premium_cent_l"] = out[
        "base_structural_local_premium_cent_l"
    ]
    out["historical_model_coverage"] = (
        out[active_coverage_cols].mean(axis=1)
        if active_coverage_cols else 0.0
    )
    out["diagnostic_model_coverage"] = (
        out[all_coverage_cols].mean(axis=1)
        if all_coverage_cols else 0.0
    )

    def _min_conf(row) -> str:
        vals = [str(row[c]) for c in active_confidence_cols]
        return min(vals, key=lambda x: CONFIDENCE_RANK.get(x, 0)) if vals else "INSUFFICIENT"

    out["structural_premium_confidence"] = out.apply(_min_conf, axis=1)

    if "accessibility_class_model_confidence" in out.columns:
        out["logistics_confidence"] = out[
            "accessibility_class_model_confidence"
        ].astype(str)
    else:
        out["logistics_confidence"] = "INSUFFICIENT"
    out["logistics_status"] = np.where(
        out["logistics_confidence"] == "INSUFFICIENT",
        "PROVISIONAL",
        "ESTIMATED",
    )
    out["logistics_role"] = "DIAGNOSTIC_ONLY"
    out["logistics_applied_to_fair_price"] = False
    out["road_type_applied_to_fair_price"] = "road_type" in active_names
    return out

def apply_historical_local_model(
    scored: pd.DataFrame,
    *,
    model: pd.DataFrame,
    meta: dict,
    logistics_overrides: pd.DataFrame | None = None,
    persistent_bias: pd.DataFrame | None = None,
    local_config: LocalAdjustmentConfig = LocalAdjustmentConfig(),
    logistics_config: LogisticsConfig = LogisticsConfig(),
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()

    fuel_values = scored["fuel"].dropna().astype(str).unique()
    if len(fuel_values) != 1:
        raise ValueError("apply_historical_local_model expects one fuel at a time")
    fuel = fuel_values[0]

    geo_cols = {
        "nearest_station_km",
        "stations_within_10km",
        "competition_bucket",
        "isolation_bucket",
    }
    with_geo = (
        scored.copy()
        if geo_cols.issubset(scored.columns)
        else add_competition_features(scored)
    )
    accessible = add_accessibility_features(
        with_geo,
        overrides=logistics_overrides,
        config=logistics_config,
    )

    out = predict_structural_premium(
        accessible,
        fuel=fuel,
        model=model,
        meta=meta,
    )
    out["local_fair_price_eur_l"] = (
        out["fair_price_eur_l"]
        + out["structural_local_premium_cent_l"] / 100.0
    )
    out["local_residual_cent_l"] = 100.0 * (
        out["price_eur_l"] - out["local_fair_price_eur_l"]
    )

    if persistent_bias is not None and not persistent_bias.empty:
        pb = persistent_bias[
            persistent_bias["fuel"].astype(str) == str(fuel)
        ].copy()
        pb["station_id"] = pb["station_id"].astype(str)
        out["_station_id_key"] = out["id"].astype(str)
        keep = [
            "station_id",
            "persistent_station_bias_cent_l",
            "persistent_station_bias_raw_cent_l",
            "persistent_bias_months",
            "persistent_bias_observations",
            "persistent_bias_confidence",
            "persistence_status",
        ]
        keep = [c for c in keep if c in pb.columns]
        out = out.merge(
            pb[keep],
            left_on="_station_id_key",
            right_on="station_id",
            how="left",
        )
        out = out.drop(columns=["_station_id_key", "station_id"], errors="ignore")

    out["local_model_type"] = "HISTORICAL_PANEL"
    fmeta = meta["fuels"][str(fuel)]
    out["local_model_train_start"] = fmeta["train_start"]
    out["local_model_train_end"] = fmeta["train_end"]
    out["local_model_months"] = int(fmeta["n_months"])

    final = add_local_peer_scores(out, config=local_config)

    if "market_freshness" in final.columns:
        stale = final["market_freshness"].astype(str) == "VERY_STALE"
        final.loc[stale, "local_flag"] = "MARKET_DATA_STALE"

    return final.sort_values(
        ["local_anomaly_score", "local_residual_cent_l"],
        ascending=False,
        na_position="last",
    )


def validate_historical_model(
    panel: pd.DataFrame,
    *,
    holdout_months: int = 3,
    config: HistoricalLocalModelConfig = HistoricalLocalModelConfig(),
    factors: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    records = []
    for fuel, fp in panel.groupby("fuel", sort=True):
        dates = sorted(pd.to_datetime(fp["date"]).unique())
        if len(dates) <= holdout_months + 3:
            continue
        test_dates = dates[-holdout_months:]
        train = fp[~pd.to_datetime(fp["date"]).isin(test_dates)].copy()
        test = fp[pd.to_datetime(fp["date"]).isin(test_dates)].copy()

        model, meta, _ = fit_historical_local_model(train, config=config, factors=factors)
        pred = predict_structural_premium(
            test,
            fuel=str(fuel),
            model=model,
            meta=meta,
        )
        y = test["relative_price_cent_l"].to_numpy(float)
        yhat = pred["structural_local_premium_cent_l"].to_numpy(float)

        records.append(
            {
                "fuel": str(fuel),
                "train_end": str(pd.Timestamp(train["date"].max()).date()),
                "test_start": str(pd.Timestamp(test["date"].min()).date()),
                "test_end": str(pd.Timestamp(test["date"].max()).date()),
                "n_test": int(len(test)),
                "baseline_mae_cent_l": float(np.mean(np.abs(y))),
                "model_mae_cent_l": float(np.mean(np.abs(y - yhat))),
                "baseline_rmse_cent_l": float(np.sqrt(np.mean(y ** 2))),
                "model_rmse_cent_l": float(np.sqrt(np.mean((y - yhat) ** 2))),
            }
        )

    out = pd.DataFrame(records)
    if not out.empty:
        out["mae_improvement_pct"] = 100.0 * (
            1.0 - out["model_mae_cent_l"] / out["baseline_mae_cent_l"]
        )
        out["rmse_improvement_pct"] = 100.0 * (
            1.0 - out["model_rmse_cent_l"] / out["baseline_rmse_cent_l"]
        )
    return out



def _score_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    fuel: str,
    split_name: str,
    config: HistoricalLocalModelConfig,
    factors: tuple[str, ...] | list[str] | None = None,
) -> dict | None:
    if train.empty or test.empty:
        return None
    model, meta, _ = fit_historical_local_model(train, config=config, factors=factors)
    pred = predict_structural_premium(test, fuel=str(fuel), model=model, meta=meta)
    y = test["relative_price_cent_l"].to_numpy(float)
    yhat = pred["structural_local_premium_cent_l"].to_numpy(float)
    baseline_mae = float(np.mean(np.abs(y)))
    model_mae = float(np.mean(np.abs(y - yhat)))
    baseline_rmse = float(np.sqrt(np.mean(y ** 2)))
    model_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {
        "fuel": str(fuel),
        "split": split_name,
        "train_start": str(pd.Timestamp(train["date"].min()).date()),
        "train_end": str(pd.Timestamp(train["date"].max()).date()),
        "test_start": str(pd.Timestamp(test["date"].min()).date()),
        "test_end": str(pd.Timestamp(test["date"].max()).date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "baseline_mae_cent_l": baseline_mae,
        "model_mae_cent_l": model_mae,
        "baseline_rmse_cent_l": baseline_rmse,
        "model_rmse_cent_l": model_rmse,
        "mae_improvement_pct": 100.0 * (1.0 - model_mae / baseline_mae),
        "rmse_improvement_pct": 100.0 * (1.0 - model_rmse / baseline_rmse),
    }


def validate_regime_splits(
    panel: pd.DataFrame,
    *,
    shock_start: str | pd.Timestamp = "2026-02-28",
    pre_shock_holdout_months: int = 3,
    transition_months: int = 4,
    current_holdout_months: int = 3,
    config: HistoricalLocalModelConfig = HistoricalLocalModelConfig(),
) -> pd.DataFrame:
    """Chronological validation around the February 2026 energy-regime break.

    Splits are leakage-free:
    - PRE_SHOCK: last N pre-shock months, trained only on earlier history.
    - SHOCK_ONSET: first transition months starting at/after shock_start,
      trained only on pre-shock history.
    - CURRENT_REGIME: following months, trained through the transition period.
    """
    shock_start = pd.Timestamp(shock_start)
    records: list[dict] = []
    for fuel, fp0 in panel.groupby("fuel", sort=True):
        fp = fp0.copy()
        fp["date"] = pd.to_datetime(fp["date"])
        dates = sorted(fp["date"].unique())
        pre_dates = [d for d in dates if pd.Timestamp(d) < shock_start]
        post_dates = [d for d in dates if pd.Timestamp(d) >= shock_start]

        if len(pre_dates) > pre_shock_holdout_months:
            test_dates = pre_dates[-pre_shock_holdout_months:]
            train_dates = pre_dates[:-pre_shock_holdout_months]
            rec = _score_split(
                fp[fp["date"].isin(train_dates)],
                fp[fp["date"].isin(test_dates)],
                fuel=str(fuel), split_name="PRE_SHOCK",
                config=config,
            )
            if rec: records.append(rec)

        onset_dates = post_dates[:transition_months]
        if pre_dates and onset_dates:
            rec = _score_split(
                fp[fp["date"].isin(pre_dates)],
                fp[fp["date"].isin(onset_dates)],
                fuel=str(fuel), split_name="SHOCK_ONSET",
                config=config,
            )
            if rec: records.append(rec)

        current_dates = post_dates[transition_months:transition_months + current_holdout_months]
        if current_dates:
            first_current = pd.Timestamp(current_dates[0])
            train = fp[fp["date"] < first_current]
            test = fp[fp["date"].isin(current_dates)]
            rec = _score_split(
                train, test,
                fuel=str(fuel), split_name="CURRENT_REGIME",
                config=config,
            )
            if rec: records.append(rec)

    return pd.DataFrame(records)


def validate_factor_ablation(
    panel: pd.DataFrame,
    *,
    holdout_months: int = 3,
    config: HistoricalLocalModelConfig = HistoricalLocalModelConfig(),
) -> pd.DataFrame:
    """Progressive out-of-sample ablation of structural factors."""
    factor_order = [spec[0] for spec in FACTOR_SPECS]
    rows: list[dict] = []
    for fuel, fp in panel.groupby("fuel", sort=True):
        dates = sorted(pd.to_datetime(fp["date"]).unique())
        if len(dates) <= holdout_months + 3:
            continue
        test_dates = dates[-holdout_months:]
        train = fp[~pd.to_datetime(fp["date"]).isin(test_dates)].copy()
        test = fp[pd.to_datetime(fp["date"]).isin(test_dates)].copy()
        y = test["relative_price_cent_l"].to_numpy(float)
        baseline_mae = float(np.mean(np.abs(y)))
        baseline_rmse = float(np.sqrt(np.mean(y ** 2)))
        rows.append({
            "fuel": str(fuel), "step": 0, "model": "baseline",
            "factors": "", "n_test": int(len(test)),
            "mae_cent_l": baseline_mae, "rmse_cent_l": baseline_rmse,
            "mae_improvement_vs_baseline_pct": 0.0,
            "rmse_improvement_vs_baseline_pct": 0.0,
        })
        selected: list[str] = []
        for step, factor in enumerate(factor_order, start=1):
            selected.append(factor)
            model, meta, _ = fit_historical_local_model(
                train, config=config, factors=tuple(selected)
            )
            pred = predict_structural_premium(test, fuel=str(fuel), model=model, meta=meta)
            yhat = pred["structural_local_premium_cent_l"].to_numpy(float)
            mae = float(np.mean(np.abs(y - yhat)))
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            rows.append({
                "fuel": str(fuel), "step": step,
                "model": "+".join(selected),
                "factors": ",".join(selected),
                "n_test": int(len(test)),
                "mae_cent_l": mae, "rmse_cent_l": rmse,
                "mae_improvement_vs_baseline_pct": 100.0 * (1.0 - mae / baseline_mae),
                "rmse_improvement_vs_baseline_pct": 100.0 * (1.0 - rmse / baseline_rmse),
            })
    return pd.DataFrame(rows)

def save_historical_model(
    model: pd.DataFrame,
    meta: dict,
    persistent_bias: pd.DataFrame,
    *,
    model_path: str | Path,
    meta_path: str | Path,
    persistent_path: str | Path,
) -> None:
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    persistent_path = Path(persistent_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    persistent_path.parent.mkdir(parents=True, exist_ok=True)

    model.to_csv(model_path, index=False)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    persistent_bias.to_csv(persistent_path, index=False)


def load_historical_model(
    model_path: str | Path,
    meta_path: str | Path,
    persistent_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    model = pd.read_csv(model_path, dtype={"level": "string", "fuel": "string"})
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if persistent_path is not None and Path(persistent_path).exists():
        persistent = pd.read_csv(
            persistent_path,
            dtype={"station_id": "string", "fuel": "string"},
        )
    else:
        persistent = pd.DataFrame()
    return model, meta, persistent
