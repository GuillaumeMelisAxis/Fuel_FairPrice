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


def _confidence_from_unique_stations(n: int, config: HistoricalLocalModelConfig) -> str:
    if n >= config.high_confidence_stations:
        return "HIGH"
    if n >= config.medium_confidence_stations:
        return "MEDIUM"
    if n >= config.low_confidence_stations:
        return "LOW"
    return "INSUFFICIENT"


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
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit v0.5 structural local premiums on a station-date panel.

    The common date/fuel price level is removed beforehand by
    ``relative_price_cent_l``. The model therefore estimates persistent observable
    local structure rather than the national market level.

    A station-specific persistent bias is estimated separately for diagnostics and
    is deliberately NOT included in the fair-price premium.
    """
    if panel.empty:
        raise ValueError("Historical panel is empty")

    required = {
        "station_id",
        "date",
        "fuel",
        "relative_price_cent_l",
        "relative_price_robust_score",
    }
    required.update(spec[0] for spec in FACTOR_SPECS)
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
        work = fuel_panel.copy()
        for factor, _, _ in FACTOR_SPECS:
            work[factor] = work[factor].fillna("UNKNOWN").astype(str)

        train = work[
            work["relative_price_robust_score"].abs()
            <= float(config.trim_abs_robust_score)
        ].copy()
        if train.empty:
            train = work.copy()

        effect_maps: dict[str, dict[str, float]] = {}
        effect_series = {
            effect_name: pd.Series(0.0, index=train.index)
            for _, effect_name, _ in FACTOR_SPECS
        }
        stats_by_effect: dict[str, pd.DataFrame] = {}

        for _ in range(config.n_backfit_iterations):
            for factor, effect_name, shrink_attr in FACTOR_SPECS:
                other = sum(
                    (
                        values
                        for name, values in effect_series.items()
                        if name != effect_name
                    ),
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
                effect_maps[effect_name] = mapping
                stats_by_effect[effect_name] = stats
                effect_series[effect_name] = _map_effect(train[factor], mapping)

        train_pred_raw = sum(
            effect_series.values(),
            start=pd.Series(0.0, index=train.index),
        )
        centre = float(train_pred_raw.median())

        # Export factor levels and reliability diagnostics.
        for factor, effect_name, shrink_attr in FACTOR_SPECS:
            stats = stats_by_effect[effect_name].copy()
            for rec in stats.to_dict("records"):
                n_stations = int(rec["n_stations"])
                rows.append(
                    {
                        "fuel": str(fuel),
                        "factor": factor,
                        "effect_column": effect_name,
                        "level": str(rec[factor]),
                        "effect_cent_l": float(rec["effect_cent_l"]),
                        "raw_effect_cent_l": float(rec["raw_effect_cent_l"]),
                        "n_obs": int(rec["n_obs"]),
                        "n_stations": n_stations,
                        "confidence": _confidence_from_unique_stations(
                            n_stations, config
                        ),
                        "shrinkage_stations": float(getattr(config, shrink_attr)),
                    }
                )

        # Persistent station bias: diagnostic only.
        pred_all_raw = pd.Series(0.0, index=work.index)
        for factor, effect_name, _ in FACTOR_SPECS:
            pred_all_raw += _map_effect(work[factor], effect_maps[effect_name])
        pred_all = pred_all_raw - centre
        work["_structural_prediction"] = pred_all
        work["_post_model_residual"] = (
            work["relative_price_cent_l"] - work["_structural_prediction"]
        )

        persistent = (
            work.groupby("station_id", as_index=False)
            .agg(
                persistent_station_bias_raw_cent_l=(
                    "_post_model_residual",
                    "median",
                ),
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
        persistent["fuel"] = str(fuel)

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

        fuel_meta[str(fuel)] = {
            "centre_cent_l": centre,
            "train_start": str(pd.Timestamp(train["date"].min()).date()),
            "train_end": str(pd.Timestamp(train["date"].max()).date()),
            "n_observations": int(len(train)),
            "n_unique_stations": int(train["station_id"].nunique()),
            "n_months": int(train["date"].nunique()),
        }

    model = pd.DataFrame(rows)
    persistent_bias = (
        pd.concat(persistent_frames, ignore_index=True)
        if persistent_frames
        else pd.DataFrame()
    )
    meta = {
        "model_version": "0.5.0",
        "model_type": "HISTORICAL_PANEL",
        "target": "relative_price_cent_l",
        "time_effect": "date_x_fuel_national_station_median_removed",
        "station_fixed_effect_applied_to_fair_price": False,
        "factors": [spec[0] for spec in FACTOR_SPECS],
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
    maps, confidence_maps = _model_maps_for_fuel(model, fuel)
    fuel_meta = meta["fuels"][str(fuel)]
    centre = float(fuel_meta["centre_cent_l"])

    nonlog_cols = []
    coverage_cols = []
    confidence_cols = []

    for factor, effect_name, _ in FACTOR_SPECS:
        values = out.get(factor, pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN").astype(str)
        mapping = maps.get(factor, {})
        conf_mapping = confidence_maps.get(factor, {})

        out[effect_name] = values.map(mapping).fillna(0.0).astype(float)
        matched = values.isin(mapping.keys())
        coverage_col = f"{factor}_model_match"
        out[coverage_col] = matched
        coverage_cols.append(coverage_col)

        ccol = f"{factor}_model_confidence"
        out[ccol] = values.map(conf_mapping).fillna("INSUFFICIENT")
        confidence_cols.append(ccol)

        if factor != "accessibility_class":
            nonlog_cols.append(effect_name)

    out["base_structural_local_premium_cent_l"] = (
        out[nonlog_cols].sum(axis=1) - centre
    )
    out["structural_local_premium_cent_l"] = (
        out["base_structural_local_premium_cent_l"]
        + out["logistics_premium_cent_l"]
    )
    out["historical_model_coverage"] = out[coverage_cols].mean(axis=1)

    def _min_conf(row) -> str:
        vals = [str(row[c]) for c in confidence_cols]
        return min(vals, key=lambda x: CONFIDENCE_RANK.get(x, 0))

    out["structural_premium_confidence"] = out.apply(_min_conf, axis=1)
    out["logistics_confidence"] = out["accessibility_class_model_confidence"].astype(str)
    out["logistics_status"] = np.where(
        out["logistics_confidence"] == "INSUFFICIENT",
        "PROVISIONAL",
        "ESTIMATED",
    )
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
) -> pd.DataFrame:
    records = []
    for fuel, fp in panel.groupby("fuel", sort=True):
        dates = sorted(pd.to_datetime(fp["date"]).unique())
        if len(dates) <= holdout_months + 3:
            continue
        test_dates = dates[-holdout_months:]
        train = fp[~pd.to_datetime(fp["date"]).isin(test_dates)].copy()
        test = fp[pd.to_datetime(fp["date"]).isin(test_dates)].copy()

        model, meta, _ = fit_historical_local_model(train, config=config)
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
