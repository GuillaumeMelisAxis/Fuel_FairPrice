from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fuel_fair_price.data.geography import add_admin_codes
from fuel_fair_price.models.anomaly import MAD_NORMALIZER
from fuel_fair_price.models.local_adjustment import add_competition_features
from fuel_fair_price.models.logistics import (
    LogisticsConfig,
    add_accessibility_features,
)


@dataclass(frozen=True)
class HistoricalPanelConfig:
    max_price_age_days: float = 30.0
    min_price_eur_l: float = 0.5
    max_price_eur_l: float = 4.0
    robust_scale_floor_cent_l: float = 1.0


def month_end_dates(start, end) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end).to_period("M").to_timestamp("M")
    if end < start:
        return pd.DatetimeIndex([])
    return pd.date_range(start, end, freq="ME")


def build_monthly_station_panel(
    events: pd.DataFrame,
    *,
    start,
    end,
    logistics_overrides: pd.DataFrame | None = None,
    config: HistoricalPanelConfig = HistoricalPanelConfig(),
    logistics_config: LogisticsConfig = LogisticsConfig(),
    add_geographic_features: bool = True,
) -> pd.DataFrame:
    """Reconstruct month-end station snapshots from official price-change events.

    The last known price at each month-end is kept only if it is no older than
    ``max_price_age_days``. Competition/isolation are recomputed on each historical
    network snapshot, then accessibility/logistics classes are assigned.

    The panel target is:
        relative_price_cent_l =
            100 * (station price - national median price for date/fuel)

    This removes the date x fuel common component and is equivalent to absorbing
    a date/fuel fixed effect before estimating structural local premiums.
    """
    if events.empty:
        return pd.DataFrame()

    ev = events.copy()
    if "station_id" not in ev.columns and "id" in ev.columns:
        ev["station_id"] = ev["id"].astype(str)
    required = {"station_id", "fuel", "updated_at", "price_eur_l"}
    missing = required - set(ev.columns)
    if missing:
        raise KeyError(f"Missing event columns: {sorted(missing)}")

    ev["station_id"] = ev["station_id"].astype(str)
    ev["updated_at"] = pd.to_datetime(ev["updated_at"], errors="coerce")
    ev["price_eur_l"] = pd.to_numeric(ev["price_eur_l"], errors="coerce")
    ev = ev.dropna(subset=["updated_at", "price_eur_l"]).copy()
    ev = ev[
        ev["price_eur_l"].between(config.min_price_eur_l, config.max_price_eur_l)
    ].copy()
    ev = ev.sort_values(["station_id", "fuel", "updated_at"])

    outputs: list[pd.DataFrame] = []
    for snap_date in month_end_dates(start, end):
        cutoff = snap_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        eligible = ev[ev["updated_at"] <= cutoff]
        if eligible.empty:
            continue

        snap = eligible.groupby(["station_id", "fuel"], as_index=False).tail(1).copy()
        snap["date"] = pd.Timestamp(snap_date).normalize()
        snap["price_age_days"] = (
            cutoff - snap["updated_at"]
        ).dt.total_seconds() / 86400.0
        snap = snap[
            (snap["price_age_days"] >= 0)
            & (snap["price_age_days"] <= config.max_price_age_days)
        ].copy()
        if snap.empty:
            continue

        snap["id"] = snap["station_id"].astype(str)
        snap = add_admin_codes(snap)

        if add_geographic_features:
            snap = add_competition_features(snap)
            snap = add_accessibility_features(
                snap,
                overrides=logistics_overrides,
                config=logistics_config,
            )

        med = snap.groupby("fuel")["price_eur_l"].transform("median")
        snap["national_station_median_eur_l"] = med
        snap["relative_price_cent_l"] = 100.0 * (snap["price_eur_l"] - med)

        def _robust_group_score(s: pd.Series) -> pd.Series:
            vals = pd.to_numeric(s, errors="coerce")
            m = float(vals.median())
            mad = float((vals - m).abs().median())
            scale = max(
                MAD_NORMALIZER * mad,
                float(config.robust_scale_floor_cent_l),
            )
            return (vals - m) / scale

        snap["relative_price_robust_score"] = (
            snap.groupby("fuel", group_keys=False)["relative_price_cent_l"]
            .transform(_robust_group_score)
        )
        outputs.append(snap)

    if not outputs:
        return pd.DataFrame()

    panel = pd.concat(outputs, ignore_index=True)
    panel = panel.sort_values(["date", "fuel", "station_id"]).reset_index(drop=True)
    return panel
