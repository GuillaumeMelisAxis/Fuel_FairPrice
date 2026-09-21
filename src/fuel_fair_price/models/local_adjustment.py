from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fuel_fair_price.models.anomaly import MAD_NORMALIZER, robust_zscore
from fuel_fair_price.models.logistics import (
    LogisticsConfig,
    add_accessibility_features,
    fit_logistics_premium,
)

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class LocalAdjustmentConfig:
    min_peer_count: int = 15
    peer_radii_km: tuple[float, ...] = (10.0, 20.0, 50.0, 100.0)
    robust_scale_floor_cent_l: float = 2.0
    trim_abs_z: float = 3.0
    n_backfit_iterations: int = 8
    region_shrinkage: float = 40.0
    department_shrinkage: float = 25.0
    road_shrinkage: float = 40.0
    competition_shrinkage: float = 40.0
    isolation_shrinkage: float = 25.0
    min_logistics_peer_count: int = 5


def _normalise_coord(series: pd.Series, *, is_latitude: bool) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    limit = 90.0 if is_latitude else 180.0
    scaled = out.copy()
    mask = out.abs() > limit
    scaled.loc[mask] = out.loc[mask] / 100000.0
    return scaled


def add_coordinate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "latitude" not in out.columns or "longitude" not in out.columns:
        raise KeyError("latitude/longitude columns are required for local adjustment")

    out["latitude_deg"] = _normalise_coord(out["latitude"], is_latitude=True)
    out["longitude_deg"] = _normalise_coord(out["longitude"], is_latitude=False)

    valid = (
        out["latitude_deg"].between(-90, 90)
        & out["longitude_deg"].between(-180, 180)
    )
    out.loc[~valid, ["latitude_deg", "longitude_deg"]] = np.nan
    return out


def _distance_block_km(lat_block, lon_block, lat_all, lon_all):
    dlat = lat_all[None, :] - lat_block[:, None]
    dlon = lon_all[None, :] - lon_block[:, None]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_block[:, None])
        * np.cos(lat_all[None, :])
        * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_competition_features(
    stations: pd.DataFrame,
    *,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Add fuel-specific geographic competition/isolation features.

    Counts are based on stations selling the same fuel. Distances are great-circle
    distances, not driving distances. The implementation is chunked to avoid an
    NxN matrix in memory.
    """
    out = add_coordinate_columns(stations)
    pieces = []

    for fuel, group in out.groupby("fuel", dropna=False, sort=False):
        g = group.copy()
        valid_mask = g["latitude_deg"].notna() & g["longitude_deg"].notna()
        valid_idx = np.flatnonzero(valid_mask.to_numpy())

        nearest = np.full(len(g), np.nan)
        counts = {
            radius: np.full(len(g), np.nan)
            for radius in (2.0, 5.0, 10.0, 20.0, 50.0)
        }

        if len(valid_idx) >= 2:
            lat = np.radians(g.iloc[valid_idx]["latitude_deg"].to_numpy(float))
            lon = np.radians(g.iloc[valid_idx]["longitude_deg"].to_numpy(float))
            n = len(valid_idx)

            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                dist = _distance_block_km(lat[start:stop], lon[start:stop], lat, lon)
                rows = np.arange(stop - start)
                cols = np.arange(start, stop)
                dist[rows, cols] = np.inf

                target_positions = valid_idx[start:stop]
                nearest[target_positions] = np.min(dist, axis=1)
                for radius in counts:
                    counts[radius][target_positions] = np.sum(dist <= radius, axis=1)

        g["nearest_station_km"] = nearest
        for radius, values in counts.items():
            label = int(radius)
            g[f"stations_within_{label}km"] = values

        g["competition_bucket"] = pd.cut(
            g["stations_within_10km"],
            bins=[-np.inf, 2, 5, 15, np.inf],
            labels=["VERY_LOW", "LOW", "MEDIUM", "HIGH"],
        ).astype("string")

        g["isolation_bucket"] = pd.cut(
            g["nearest_station_km"],
            bins=[-np.inf, 1.0, 3.0, 8.0, np.inf],
            labels=["DENSE", "LOCAL", "ISOLATED", "VERY_ISOLATED"],
        ).astype("string")

        g["isolated_market"] = (
            (g["nearest_station_km"] > 8.0)
            | (g["stations_within_20km"] <= 2)
        )
        pieces.append(g)

    return pd.concat(pieces, ignore_index=False).sort_index()


def _shrunk_group_effect(
    frame: pd.DataFrame,
    *,
    group_col: str,
    residual_col: str,
    shrinkage: float,
) -> dict[str, float]:
    stats = (
        frame.groupby(group_col, dropna=False)[residual_col]
        .agg(["median", "count"])
        .reset_index()
    )
    stats["effect"] = stats["median"] * stats["count"] / (stats["count"] + shrinkage)
    return dict(zip(stats[group_col].astype(str), stats["effect"].astype(float)))


def _map_effect(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    return series.astype(str).map(mapping).fillna(0.0).astype(float)


def fit_structural_local_premium(
    scored: pd.DataFrame,
    *,
    config: LocalAdjustmentConfig = LocalAdjustmentConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate an interpretable, robust additive local premium.

    The target is the station fundamental spread after removing the national
    median spread for that fuel. Therefore the model does NOT absorb a national
    market-wide deviation from the fundamental fair price.

    Additive factors are learned by robust median backfitting with empirical
    shrinkage:
      region + department + road type + competition + isolation.
    """
    out = scored.copy()
    if "spread_cent_l" not in out.columns:
        raise KeyError("spread_cent_l is required before local adjustment")

    required = [
        "code_region",
        "code_departement",
        "road_type",
        "competition_bucket",
        "isolation_bucket",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)

    national_median = float(out["spread_cent_l"].median())
    out["relative_spread_cent_l"] = out["spread_cent_l"] - national_median
    out["relative_spread_z"] = robust_zscore(out["relative_spread_cent_l"])

    train = out[out["relative_spread_z"].abs() <= config.trim_abs_z].copy()
    if train.empty:
        train = out.copy()

    factors = [
        ("code_region", config.region_shrinkage, "region_effect_cent_l"),
        ("code_departement", config.department_shrinkage, "department_effect_cent_l"),
        ("road_type", config.road_shrinkage, "road_effect_cent_l"),
        ("competition_bucket", config.competition_shrinkage, "competition_effect_cent_l"),
        ("isolation_bucket", config.isolation_shrinkage, "isolation_effect_cent_l"),
    ]

    effect_maps: dict[str, dict[str, float]] = {name: {} for _, _, name in factors}
    train_effects = {name: pd.Series(0.0, index=train.index) for _, _, name in factors}

    for _ in range(config.n_backfit_iterations):
        for group_col, shrinkage, effect_name in factors:
            other = sum(
                (series for name, series in train_effects.items() if name != effect_name),
                start=pd.Series(0.0, index=train.index),
            )
            residual = train["relative_spread_cent_l"] - other
            work = train[[group_col]].copy()
            work["_residual"] = residual
            mapping = _shrunk_group_effect(
                work,
                group_col=group_col,
                residual_col="_residual",
                shrinkage=shrinkage,
            )
            effect_maps[effect_name] = mapping
            train_effects[effect_name] = _map_effect(train[group_col], mapping)

    for group_col, _, effect_name in factors:
        out[effect_name] = _map_effect(out[group_col], effect_maps[effect_name])

    effect_cols = [name for _, _, name in factors]
    out["structural_local_premium_cent_l"] = out[effect_cols].sum(axis=1)

    # Re-center to keep the national fair price unchanged in aggregate.
    centre = float(out["structural_local_premium_cent_l"].median())
    out["structural_local_premium_cent_l"] -= centre

    out["local_fair_price_eur_l"] = (
        out["fair_price_eur_l"] + out["structural_local_premium_cent_l"] / 100.0
    )
    out["local_residual_cent_l"] = 100.0 * (
        out["price_eur_l"] - out["local_fair_price_eur_l"]
    )

    diagnostics = []
    for group_col, _, effect_name in factors:
        for level, effect in effect_maps[effect_name].items():
            count = int((train[group_col].astype(str) == level).sum())
            diagnostics.append(
                {
                    "factor": group_col,
                    "level": level,
                    "effect_cent_l": float(effect),
                    "train_count": count,
                }
            )

    diag = pd.DataFrame(diagnostics)
    if not diag.empty:
        diag["fuel"] = str(out["fuel"].iloc[0]) if "fuel" in out.columns and not out.empty else "UNKNOWN"
        diag["national_median_spread_cent_l"] = national_median

    return out, diag


def add_local_peer_scores(
    adjusted: pd.DataFrame,
    *,
    config: LocalAdjustmentConfig = LocalAdjustmentConfig(),
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Compute the stabilized local anomaly score.

    Public v0.4.2 terminology deliberately avoids the name ``z-score`` because
    the denominator is a robust peer scale with an explicit 2 c/L floor, not a
    Gaussian standard deviation.

        local_excess = local_residual - median(peer residuals)
        scale = max(1.4826 * MAD(peer residuals), scale_floor)
        local_anomaly_score = local_excess / scale

    Logistics-aware peer selection:
      * FERRY_ISLAND stations are never compared directly with mainland stations.
        Same-island peers are preferred; if too few exist, other ferry-island
        stations form a deliberately LOW-confidence logistics peer group.
      * Other accessibility classes retain the adaptive geographic peer search.
    """
    out = add_coordinate_columns(adjusted)
    out["local_peer_count"] = 0
    out["local_peer_radius_km"] = np.nan
    out["local_peer_median_residual_cent_l"] = np.nan
    out["local_peer_mad_cent_l"] = np.nan
    out["local_peer_scale_cent_l"] = np.nan
    out["local_excess_cent_l"] = np.nan
    out["local_anomaly_score"] = np.nan
    out["peer_selection_method"] = "NONE"
    out["peer_confidence"] = "INSUFFICIENT"

    for col, default in (
        ("accessibility_class", "MAINLAND"),
        ("logistics_peer_group", "MAINLAND"),
    ):
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default).astype(str)

    valid_mask = (
        out["latitude_deg"].notna()
        & out["longitude_deg"].notna()
        & out["local_residual_cent_l"].notna()
    )
    valid_idx = np.flatnonzero(valid_mask.to_numpy())

    if len(valid_idx) >= 2:
        valid = out.iloc[valid_idx].copy()
        lat = np.radians(valid["latitude_deg"].to_numpy(float))
        lon = np.radians(valid["longitude_deg"].to_numpy(float))
        residuals = valid["local_residual_cent_l"].to_numpy(float)
        access = valid["accessibility_class"].astype(str).to_numpy()
        logistics_group = valid["logistics_peer_group"].astype(str).to_numpy()
        n = len(valid_idx)

        peer_count = np.zeros(n, dtype=int)
        peer_radius = np.full(n, np.nan)
        peer_median = np.full(n, np.nan)
        peer_mad = np.full(n, np.nan)
        peer_scale = np.full(n, np.nan)
        local_excess = np.full(n, np.nan)
        anomaly_score = np.full(n, np.nan)
        selection_method = np.full(n, "NONE", dtype=object)
        confidence = np.full(n, "INSUFFICIENT", dtype=object)

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            dist = _distance_block_km(lat[start:stop], lon[start:stop], lat, lon)
            rows = np.arange(stop - start)
            cols = np.arange(start, stop)
            dist[rows, cols] = np.inf

            for local_row in range(stop - start):
                global_pos = start + local_row
                selected = None
                selected_radius = np.nan
                method = "NONE"
                conf_override = None

                # A ferry-dependent island is not economically adjacent to the
                # mainland simply because the coastline is nearby.
                if access[global_pos] == "FERRY_ISLAND":
                    same_group = (
                        (logistics_group == logistics_group[global_pos])
                        & (np.arange(n) != global_pos)
                    )
                    if int(same_group.sum()) >= config.min_logistics_peer_count:
                        selected = same_group
                        method = "SAME_LOGISTICS_GROUP"
                        conf_override = "MEDIUM" if int(same_group.sum()) >= config.min_peer_count else "LOW"
                    else:
                        same_class = (
                            (access == "FERRY_ISLAND")
                            & (np.arange(n) != global_pos)
                        )
                        if int(same_class.sum()) >= config.min_logistics_peer_count:
                            selected = same_class
                            method = "FERRY_CLASS"
                            conf_override = "LOW"

                # Standard adaptive geographic peer search for mainland,
                # road-connected islands and remote/isolated mainland markets.
                if selected is None and access[global_pos] != "FERRY_ISLAND":
                    for radius in config.peer_radii_km:
                        mask = dist[local_row] <= radius
                        if int(mask.sum()) >= config.min_peer_count:
                            selected = mask
                            selected_radius = float(radius)
                            method = "RADIUS"
                            break

                    if selected is None:
                        finite_order = np.argsort(dist[local_row])
                        finite_order = finite_order[np.isfinite(dist[local_row][finite_order])]
                        if len(finite_order) >= config.min_peer_count:
                            chosen = finite_order[: config.min_peer_count]
                            selected = np.zeros(n, dtype=bool)
                            selected[chosen] = True
                            selected_radius = float(np.max(dist[local_row][chosen]))
                            method = "NEAREST_N"

                if selected is None:
                    continue

                min_required = (
                    config.min_logistics_peer_count
                    if method in {"SAME_LOGISTICS_GROUP", "FERRY_CLASS"}
                    else config.min_peer_count
                )
                if int(selected.sum()) < min_required:
                    continue

                vals = residuals[selected]
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)))
                scale = max(
                    MAD_NORMALIZER * mad,
                    float(config.robust_scale_floor_cent_l),
                )
                excess = float(residuals[global_pos] - med)
                score = float(excess / scale)
                count = int(selected.sum())

                if conf_override is not None:
                    conf = conf_override
                elif method == "RADIUS" and selected_radius <= 20.0 and count >= 30:
                    conf = "HIGH"
                elif method == "RADIUS" and selected_radius <= 50.0 and count >= config.min_peer_count:
                    conf = "MEDIUM"
                else:
                    conf = "LOW"

                peer_count[global_pos] = count
                peer_radius[global_pos] = selected_radius
                peer_median[global_pos] = med
                peer_mad[global_pos] = mad
                peer_scale[global_pos] = scale
                local_excess[global_pos] = excess
                anomaly_score[global_pos] = score
                selection_method[global_pos] = method
                confidence[global_pos] = conf

        assignments = {
            "local_peer_count": peer_count,
            "local_peer_radius_km": peer_radius,
            "local_peer_median_residual_cent_l": peer_median,
            "local_peer_mad_cent_l": peer_mad,
            "local_peer_scale_cent_l": peer_scale,
            "local_excess_cent_l": local_excess,
            "local_anomaly_score": anomaly_score,
            "peer_selection_method": selection_method,
            "peer_confidence": confidence,
        }
        for col, values in assignments.items():
            out.iloc[valid_idx, out.columns.get_loc(col)] = values

    out["local_flag"] = "NORMAL"
    enough_confidence = out["peer_confidence"].isin(["HIGH", "MEDIUM"])
    high = (
        enough_confidence
        & (out["local_anomaly_score"] > 2.0)
        & (out["local_excess_cent_l"] > 5.0)
    )
    very_high = (
        enough_confidence
        & (out["local_anomaly_score"] > 3.0)
        & (out["local_excess_cent_l"] > 10.0)
    )
    low_conf_review = (
        (out["peer_confidence"] == "LOW")
        & (out["local_anomaly_score"] > 2.0)
        & (out["local_excess_cent_l"] > 5.0)
    )
    insufficient_peers = out["peer_confidence"] == "INSUFFICIENT"

    out.loc[high, "local_flag"] = "HIGH"
    out.loc[very_high, "local_flag"] = "VERY_HIGH"
    out.loc[low_conf_review, "local_flag"] = "REVIEW_LOW_CONFIDENCE"
    out.loc[
        insufficient_peers,
        "local_flag",
    ] = "UNASSESSED_INSUFFICIENT_PEERS"

    return out


def apply_local_adjustment(
    scored: pd.DataFrame,
    *,
    config: LocalAdjustmentConfig = LocalAdjustmentConfig(),
    logistics_overrides: pd.DataFrame | None = None,
    logistics_config: LogisticsConfig = LogisticsConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full v0.4.2 local + logistics-aware adjustment pipeline for one fuel."""
    geo_cols = {
        "nearest_station_km",
        "stations_within_10km",
        "competition_bucket",
        "isolation_bucket",
    }
    with_geo = scored.copy() if geo_cols.issubset(scored.columns) else add_competition_features(scored)

    # 1) Existing v0.4 base structural adjustment.
    structural, base_effects = fit_structural_local_premium(with_geo, config=config)

    # 2) Accessibility classification and logistics premium.
    accessible = add_accessibility_features(
        structural,
        overrides=logistics_overrides,
        config=logistics_config,
    )
    logistics_adjusted, logistics_effects = fit_logistics_premium(
        accessible,
        config=logistics_config,
    )

    # 3) Logistics-aware peer comparison and anomaly score.
    final = add_local_peer_scores(logistics_adjusted, config=config)

    if "market_freshness" in final.columns:
        stale = final["market_freshness"].astype(str) == "VERY_STALE"
        final.loc[stale, "local_flag"] = "MARKET_DATA_STALE"

    effects = pd.concat(
        [x for x in (base_effects, logistics_effects) if x is not None and not x.empty],
        ignore_index=True,
        sort=False,
    ) if (not base_effects.empty or not logistics_effects.empty) else pd.DataFrame()

    final = final.sort_values(
        ["local_anomaly_score", "local_residual_cent_l"],
        ascending=False,
        na_position="last",
    )
    return final, effects
