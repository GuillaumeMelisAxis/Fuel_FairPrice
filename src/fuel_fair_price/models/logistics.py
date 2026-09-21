from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogisticsConfig:
    shrinkage: float = 10.0
    remote_nearest_km: float = 15.0
    remote_max_stations_20km: int = 1
    high_confidence_count: int = 30
    medium_confidence_count: int = 10
    low_confidence_count: int = 3


def _normalise_text(value) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().replace("’", "'").split())


def load_logistics_overrides(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype="string")
    required = {
        "match_type",
        "match_value",
        "accessibility_class",
        "logistics_peer_group",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing logistics override columns: {sorted(missing)}")
    return df


def add_accessibility_features(
    stations: pd.DataFrame,
    *,
    overrides: pd.DataFrame | None = None,
    config: LogisticsConfig = LogisticsConfig(),
) -> pd.DataFrame:
    """Classify station accessibility without assigning arbitrary cent/L premiums.

    Explicit overrides identify known island logistics. Mainland stations that are
    extremely isolated in the observed same-fuel network are labelled
    ``REMOTE_ISOLATED``. This is deliberately not called ``REMOTE_MOUNTAIN`` because
    altitude/road-topology data are not available in the station feed.
    """
    out = stations.copy()
    out["accessibility_class"] = "MAINLAND"
    out["logistics_peer_group"] = "MAINLAND"
    out["accessibility_source"] = "DEFAULT"

    if overrides is not None and not overrides.empty:
        work = overrides.copy()
        work["_match_value_norm"] = work["match_value"].map(_normalise_text)

        if "ville" in out.columns:
            city_norm = out["ville"].map(_normalise_text)
            city_rules = work[work["match_type"].str.lower() == "ville"]
            city_map = city_rules.set_index("_match_value_norm")
            for idx, norm in city_norm.items():
                if norm and norm in city_map.index:
                    row = city_map.loc[norm]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    out.at[idx, "accessibility_class"] = str(row["accessibility_class"])
                    out.at[idx, "logistics_peer_group"] = str(row["logistics_peer_group"])
                    out.at[idx, "accessibility_source"] = "OVERRIDE_VILLE"

        if "id" in out.columns:
            id_rules = work[work["match_type"].str.lower() == "id"]
            if not id_rules.empty:
                id_map = id_rules.set_index("match_value")
                for idx, station_id in out["id"].astype(str).items():
                    if station_id in id_map.index:
                        row = id_map.loc[station_id]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        out.at[idx, "accessibility_class"] = str(row["accessibility_class"])
                        out.at[idx, "logistics_peer_group"] = str(row["logistics_peer_group"])
                        out.at[idx, "accessibility_source"] = "OVERRIDE_ID"

    # Conservative heuristic only for stations not explicitly classified.
    if {"nearest_station_km", "stations_within_20km"}.issubset(out.columns):
        remote = (
            (out["accessibility_class"] == "MAINLAND")
            & (pd.to_numeric(out["nearest_station_km"], errors="coerce") >= config.remote_nearest_km)
            & (
                pd.to_numeric(out["stations_within_20km"], errors="coerce")
                <= config.remote_max_stations_20km
            )
        )
        out.loc[remote, "accessibility_class"] = "REMOTE_ISOLATED"
        out.loc[remote, "logistics_peer_group"] = "REMOTE_ISOLATED"
        out.loc[remote, "accessibility_source"] = "ISOLATION_HEURISTIC"

    return out


def _confidence_from_count(n: int, config: LogisticsConfig) -> str:
    if n >= config.high_confidence_count:
        return "HIGH"
    if n >= config.medium_confidence_count:
        return "MEDIUM"
    if n >= config.low_confidence_count:
        return "LOW"
    return "INSUFFICIENT"


def fit_logistics_premium(
    adjusted: pd.DataFrame,
    *,
    config: LogisticsConfig = LogisticsConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a cross-sectional logistics premium with shrinkage and constraints.

    The input must already contain the base structural local residual from the v0.4
    model. ``MAINLAND`` is fixed at zero. For constrained logistics classes, a
    negative estimated effect is floored at zero: ferry/road-island/remote access is
    allowed to add cost but is not allowed to be interpreted as a logistics discount.

    This remains a v0.4.x cross-sectional estimate. v0.5 will estimate these effects
    from a historical panel.
    """
    out = adjusted.copy()
    if "local_residual_cent_l" not in out.columns:
        raise KeyError("local_residual_cent_l is required before logistics adjustment")
    if "accessibility_class" not in out.columns:
        raise KeyError("accessibility_class is required before logistics adjustment")

    out["base_structural_local_premium_cent_l"] = out[
        "structural_local_premium_cent_l"
    ].astype(float)
    out["base_local_fair_price_eur_l"] = out["local_fair_price_eur_l"].astype(float)
    out["base_local_residual_cent_l"] = out["local_residual_cent_l"].astype(float)

    mainland = out[out["accessibility_class"] == "MAINLAND"]["base_local_residual_cent_l"]
    mainland_median = float(mainland.median()) if not mainland.empty else float(
        out["base_local_residual_cent_l"].median()
    )

    diagnostics = []
    premium_map: dict[str, float] = {"MAINLAND": 0.0}
    count_map: dict[str, int] = {"MAINLAND": int((out["accessibility_class"] == "MAINLAND").sum())}
    raw_map: dict[str, float] = {"MAINLAND": 0.0}

    constrained = {"FERRY_ISLAND", "ROAD_CONNECTED_ISLAND", "REMOTE_ISOLATED"}

    for cls, group in out.groupby("accessibility_class", dropna=False):
        cls = str(cls)
        if cls == "MAINLAND":
            continue
        vals = group["base_local_residual_cent_l"].dropna()
        n = len(vals)
        if n == 0:
            raw = 0.0
        else:
            raw = float(vals.median() - mainland_median)
        shrunk = raw * n / (n + float(config.shrinkage)) if n > 0 else 0.0
        if cls in constrained:
            shrunk = max(0.0, shrunk)
        premium_map[cls] = float(shrunk)
        count_map[cls] = int(n)
        raw_map[cls] = float(raw)

    out["logistics_premium_cent_l"] = (
        out["accessibility_class"].astype(str).map(premium_map).fillna(0.0).astype(float)
    )
    out["logistics_train_count"] = (
        out["accessibility_class"].astype(str).map(count_map).fillna(0).astype(int)
    )
    out["logistics_confidence"] = out["logistics_train_count"].map(
        lambda n: _confidence_from_count(int(n), config)
    )
    out["logistics_status"] = np.where(
        out["logistics_confidence"] == "INSUFFICIENT",
        "PROVISIONAL",
        "ESTIMATED",
    )

    out["structural_local_premium_cent_l"] = (
        out["base_structural_local_premium_cent_l"] + out["logistics_premium_cent_l"]
    )
    out["local_fair_price_eur_l"] = (
        out["fair_price_eur_l"] + out["structural_local_premium_cent_l"] / 100.0
    )
    out["local_residual_cent_l"] = 100.0 * (
        out["price_eur_l"] - out["local_fair_price_eur_l"]
    )

    for cls in sorted(set(out["accessibility_class"].astype(str))):
        n_train = int(count_map.get(cls, 0))
        confidence = _confidence_from_count(n_train, config)
        diagnostics.append(
            {
                "factor": "accessibility_class",
                "level": cls,
                "effect_cent_l": float(premium_map.get(cls, 0.0)),
                "raw_effect_cent_l": float(raw_map.get(cls, 0.0)),
                "train_count": n_train,
                "confidence": confidence,
                "status": (
                    "PROVISIONAL"
                    if confidence == "INSUFFICIENT"
                    else "ESTIMATED"
                ),
            }
        )

    diag = pd.DataFrame(diagnostics)
    if not diag.empty and "fuel" in out.columns and not out.empty:
        diag["fuel"] = str(out["fuel"].iloc[0])
        diag["mainland_median_base_residual_cent_l"] = mainland_median

    return out, diag
