from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fuel_fair_price.models.historical_local import predict_structural_premium
from fuel_fair_price.models.local_adjustment import LocalAdjustmentConfig, add_local_peer_scores


@dataclass(frozen=True)
class HistoricalIndexConfig:
    base_value: float = 100.0
    min_common_stations: int = 30
    vat_rate: float = 0.20


def _normalise_station_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "station_id" not in out.columns:
        if "id" not in out.columns:
            raise KeyError("Expected station_id or id column")
        out["station_id"] = out["id"].astype("string")
    out["station_id"] = out["station_id"].astype("string")
    out["fuel"] = out["fuel"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["price_eur_l"] = pd.to_numeric(out["price_eur_l"], errors="coerce")
    return out.dropna(subset=["station_id", "fuel", "date", "price_eur_l"])


def build_chained_observed_index(
    station_prices: pd.DataFrame,
    *,
    config: HistoricalIndexConfig = HistoricalIndexConfig(),
) -> pd.DataFrame:
    """Build a composition-resistant observed pump-price index.

    Between two index dates, only stations observed on both dates determine the
    chained growth rate. This avoids jumps caused solely by stations entering or
    leaving the sample. If too few matched stations exist, the ratio of national
    medians is used transparently as a low-confidence fallback.
    """
    frame = _normalise_station_frame(station_prices)
    rows: list[dict] = []

    for fuel, fg in frame.groupby("fuel", sort=True):
        dates = sorted(fg["date"].unique())
        if not dates:
            continue

        index_level = float(config.base_value)
        previous_date = None
        previous_snapshot = None

        for pos, date in enumerate(dates):
            snap = (
                fg[fg["date"] == date][["station_id", "price_eur_l"]]
                .drop_duplicates("station_id", keep="last")
                .copy()
            )
            median_price = float(snap["price_eur_l"].median())
            station_count = int(snap["station_id"].nunique())

            if pos == 0:
                growth = 1.0
                monthly_change_pct = 0.0
                common_count = np.nan
                common_share = np.nan
                method = "BASE"
                confidence = "HIGH"
            else:
                matched = previous_snapshot.merge(
                    snap,
                    on="station_id",
                    how="inner",
                    suffixes=("_prev", "_curr"),
                )
                valid = matched[
                    (matched["price_eur_l_prev"] > 0)
                    & matched["price_eur_l_curr"].notna()
                ].copy()
                common_count = int(len(valid))
                common_share = (
                    common_count / max(1, min(len(previous_snapshot), len(snap)))
                )

                if common_count >= int(config.min_common_stations):
                    ratios = valid["price_eur_l_curr"] / valid["price_eur_l_prev"]
                    growth = float(ratios.median())
                    method = "MATCHED_STATIONS"
                    confidence = "HIGH" if common_share >= 0.70 else "MEDIUM"
                else:
                    previous_median = float(previous_snapshot["price_eur_l"].median())
                    growth = median_price / previous_median if previous_median > 0 else 1.0
                    method = "MEDIAN_FALLBACK"
                    confidence = "LOW"

                index_level *= growth
                monthly_change_pct = 100.0 * (growth - 1.0)

            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "fuel": str(fuel),
                    "observed_median_price_eur_l": median_price,
                    "observed_price_index": index_level,
                    "observed_index_change_pct": monthly_change_pct,
                    "stations_observed": station_count,
                    "matched_station_count": common_count,
                    "matched_station_share": common_share,
                    "observed_index_method": method,
                    "observed_index_confidence": confidence,
                }
            )
            previous_date = date
            previous_snapshot = snap

    return pd.DataFrame(rows).sort_values(["fuel", "date"]).reset_index(drop=True)


def build_historical_local_anomaly_rate(
    panel: pd.DataFrame,
    *,
    model: pd.DataFrame,
    meta: dict,
    local_config: LocalAdjustmentConfig = LocalAdjustmentConfig(),
) -> pd.DataFrame:
    """Reconstruct the v0.5.1 local-anomaly rate on each historical snapshot.

    This is deliberately versioned/model-dependent. The observed price index is
    model-free; this series is a retrospective diagnostic using the frozen v0.5.1
    production model.
    """
    required = {"date", "fuel", "relative_price_cent_l", "price_eur_l"}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Historical panel missing columns: {sorted(missing)}")

    rows: list[dict] = []
    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    work["fuel"] = work["fuel"].astype(str).str.upper()

    for (date, fuel), snap in work.groupby(["date", "fuel"], sort=True):
        predicted = predict_structural_premium(
            snap,
            fuel=str(fuel),
            model=model,
            meta=meta,
        )
        predicted["local_residual_cent_l"] = (
            pd.to_numeric(predicted["relative_price_cent_l"], errors="coerce")
            - pd.to_numeric(
                predicted["structural_local_premium_cent_l"], errors="coerce"
            )
        )
        scored = add_local_peer_scores(predicted, config=local_config)

        assessed = scored[scored["peer_confidence"].isin(["HIGH", "MEDIUM"])]
        anomaly = assessed[assessed["local_flag"].isin(["HIGH", "VERY_HIGH"])]
        very_high = assessed[assessed["local_flag"] == "VERY_HIGH"]

        n_assessed = int(len(assessed))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "fuel": str(fuel),
                "stations_anomaly_assessed": n_assessed,
                "local_anomaly_count": int(len(anomaly)),
                "local_very_high_count": int(len(very_high)),
                "local_anomaly_rate_pct": (
                    100.0 * len(anomaly) / n_assessed if n_assessed else np.nan
                ),
                "local_very_high_rate_pct": (
                    100.0 * len(very_high) / n_assessed if n_assessed else np.nan
                ),
                "anomaly_model_type": "HISTORICAL_PANEL_V0_5_1_RETROSPECTIVE",
            }
        )

    return pd.DataFrame(rows).sort_values(["fuel", "date"]).reset_index(drop=True)


def aggregate_live_anomaly_rate(live_scores: pd.DataFrame, index_date) -> pd.DataFrame:
    if live_scores.empty:
        return pd.DataFrame()
    work = live_scores.copy()
    work["fuel"] = work["fuel"].astype(str).str.upper()
    rows = []
    for fuel, snap in work.groupby("fuel"):
        assessed = snap[snap["peer_confidence"].isin(["HIGH", "MEDIUM"])]
        anomaly = assessed[assessed["local_flag"].isin(["HIGH", "VERY_HIGH"])]
        very_high = assessed[assessed["local_flag"] == "VERY_HIGH"]
        n_assessed = int(len(assessed))
        rows.append(
            {
                "date": pd.Timestamp(index_date).normalize(),
                "fuel": str(fuel),
                "stations_anomaly_assessed": n_assessed,
                "local_anomaly_count": int(len(anomaly)),
                "local_very_high_count": int(len(very_high)),
                "local_anomaly_rate_pct": (
                    100.0 * len(anomaly) / n_assessed if n_assessed else np.nan
                ),
                "local_very_high_rate_pct": (
                    100.0 * len(very_high) / n_assessed if n_assessed else np.nan
                ),
                "anomaly_model_type": "LIVE_V0_5_1",
            }
        )
    return pd.DataFrame(rows)


def _asof_proxy_value(group: pd.DataFrame, date: pd.Timestamp, column: str) -> float | None:
    eligible = group[group["date"] <= pd.Timestamp(date)].dropna(subset=[column])
    if eligible.empty:
        return None
    return float(eligible.iloc[-1][column])


def build_historical_refined_quote_series(
    dates_by_fuel: pd.DataFrame,
    *,
    daily_proxy: pd.DataFrame,
    anchors: pd.DataFrame,
    selected_filter: dict[str, str],
) -> pd.DataFrame:
    """Reconstruct refined-product levels with auditable DGEC anchoring.

    Official DGEC monthly values are used directly when the index date belongs to
    the same month. Missing months are reconstructed from the selected public
    product proxy relative to the nearest *past* DGEC anchor. Dates before the
    first DGEC anchor are explicit proxy backcasts from that first anchor.
    """
    targets = dates_by_fuel[["date", "fuel"]].drop_duplicates().copy()
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    targets["fuel"] = targets["fuel"].astype(str).str.upper()

    proxy = daily_proxy.copy()
    proxy["date"] = pd.to_datetime(proxy["date"]).dt.normalize()
    proxy["fuel"] = proxy["fuel"].astype(str).str.upper()

    anc = anchors.copy()
    anc["date"] = pd.to_datetime(anc["date"]).dt.normalize()
    if "effective_market_date" in anc.columns:
        anc["effective_market_date"] = pd.to_datetime(
            anc["effective_market_date"]
        ).dt.normalize()
    else:
        anc["effective_market_date"] = anc["date"]
    anc["fuel"] = anc["fuel"].astype(str).str.upper()

    rows = []
    for _, target in targets.sort_values(["fuel", "date"]).iterrows():
        fuel = str(target["fuel"])
        date = pd.Timestamp(target["date"])
        filter_col = selected_filter.get(fuel, "proxy_ma5_eur_l")
        pg = proxy[proxy["fuel"] == fuel].sort_values("date")
        ag = anc[anc["fuel"] == fuel].sort_values("effective_market_date")
        if pg.empty or ag.empty:
            rows.append(
                {
                    "date": date,
                    "fuel": fuel,
                    "refined_quote_est_eur_l": np.nan,
                    "refined_quote_source": "UNAVAILABLE",
                    "refined_anchor_date": pd.NaT,
                    "refined_anchor_eur_l": np.nan,
                    "selected_filter": filter_col,
                }
            )
            continue

        same_month = ag[ag["date"].dt.to_period("M") == date.to_period("M")]
        if not same_month.empty:
            official = same_month.iloc[-1]
            quote = float(official["refined_quote_eur_l"])
            anchor_date = pd.Timestamp(official["effective_market_date"])
            source = "DGEC_MONTHLY_OFFICIAL"
        else:
            past = ag[ag["effective_market_date"] <= date]
            if not past.empty:
                official = past.iloc[-1]
                source = "DGEC_ANCHORED_PROXY"
            else:
                official = ag.iloc[0]
                source = "DGEC_PROXY_BACKCAST"

            anchor_date = pd.Timestamp(official["effective_market_date"])
            anchor_quote = float(official["refined_quote_eur_l"])
            pv_target = _asof_proxy_value(pg, date, filter_col)
            pv_anchor = _asof_proxy_value(pg, anchor_date, filter_col)
            if pv_target is None or pv_anchor is None or pv_anchor <= 0:
                quote = np.nan
                source = "UNAVAILABLE"
            else:
                quote = anchor_quote * pv_target / pv_anchor

        rows.append(
            {
                "date": date,
                "fuel": fuel,
                "refined_quote_est_eur_l": quote,
                "refined_quote_source": source,
                "refined_anchor_date": anchor_date,
                "refined_anchor_eur_l": float(official["refined_quote_eur_l"]),
                "selected_filter": filter_col,
            }
        )

    return pd.DataFrame(rows)


def build_historical_fundamental_fair_series(
    dates_by_fuel: pd.DataFrame,
    *,
    weekly_france_prices: pd.DataFrame,
    refined_series: pd.DataFrame,
    baseline_margins: dict[str, float],
    fallback_excise: dict[str, float] | None = None,
    config: HistoricalIndexConfig = HistoricalIndexConfig(),
) -> pd.DataFrame:
    """Build a monthly historical fundamental fair-price reconstruction.

    Tax wedges are inferred from contemporaneous EU Weekly Oil Bulletin TTC/HTT
    France observations. Refined-product levels are official DGEC values when
    available and otherwise explicit DGEC-anchored proxy reconstructions.
    """
    targets = dates_by_fuel[["date", "fuel"]].drop_duplicates().copy()
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    targets["fuel"] = targets["fuel"].astype(str).str.upper()

    weekly = weekly_france_prices.copy()
    weekly["date"] = pd.to_datetime(weekly["date"]).dt.normalize()
    weekly["fuel"] = weekly["fuel"].astype(str).str.upper()
    fallback_excise = {k.upper(): float(v) for k, v in (fallback_excise or {}).items()}

    pieces = []
    for fuel, tg in targets.groupby("fuel"):
        wg = weekly[weekly["fuel"] == fuel].sort_values("date")
        left = tg.sort_values("date").copy()
        if wg.empty:
            left["eu_price_date"] = pd.NaT
            left["eu_ttc_eur_l"] = np.nan
            left["eu_htt_eur_l"] = np.nan
        else:
            right = wg[["date", "ttc_eur_l", "htt_eur_l"]].rename(
                columns={"date": "eu_price_date", "ttc_eur_l": "eu_ttc_eur_l", "htt_eur_l": "eu_htt_eur_l"}
            )
            left = pd.merge_asof(
                left.sort_values("date"),
                right.sort_values("eu_price_date"),
                left_on="date",
                right_on="eu_price_date",
                direction="backward",
                tolerance=pd.Timedelta(days=14),
            )
        pieces.append(left)

    out = pd.concat(pieces, ignore_index=True)
    out = out.merge(refined_series, on=["date", "fuel"], how="left")

    out["effective_excise_eur_l"] = (
        out["eu_ttc_eur_l"] / (1.0 + float(config.vat_rate)) - out["eu_htt_eur_l"]
    )
    out["tax_source"] = np.where(
        out["effective_excise_eur_l"].notna(),
        "EU_WEEKLY_TTC_HTT",
        "FALLBACK_CONFIG",
    )
    for fuel, excise in fallback_excise.items():
        mask = (out["fuel"] == fuel) & out["effective_excise_eur_l"].isna()
        out.loc[mask, "effective_excise_eur_l"] = float(excise)

    out["normal_distribution_margin_eur_l"] = out["fuel"].map(
        {k.upper(): float(v) for k, v in baseline_margins.items()}
    )
    out["fundamental_fair_price_eur_l"] = (
        out["refined_quote_est_eur_l"]
        + out["normal_distribution_margin_eur_l"]
        + out["effective_excise_eur_l"]
    ) * (1.0 + float(config.vat_rate))

    out["fundamental_fair_confidence"] = "MEDIUM"
    out.loc[
        out["refined_quote_source"] == "DGEC_MONTHLY_OFFICIAL",
        "fundamental_fair_confidence",
    ] = "HIGH"
    out.loc[
        out["refined_quote_source"].isin(["DGEC_PROXY_BACKCAST", "UNAVAILABLE"]),
        "fundamental_fair_confidence",
    ] = "LOW"
    out.loc[
        out["fundamental_fair_price_eur_l"].isna(),
        "fundamental_fair_confidence",
    ] = "UNAVAILABLE"

    return out.sort_values(["fuel", "date"]).reset_index(drop=True)


def combine_index_series(
    observed: pd.DataFrame,
    fair: pd.DataFrame,
    anomaly: pd.DataFrame,
    *,
    live_summary: pd.DataFrame | None = None,
    live_market_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = observed.merge(fair, on=["date", "fuel"], how="left")
    out = out.merge(anomaly, on=["date", "fuel"], how="left")

    if live_summary is not None and not live_summary.empty:
        live = live_summary.copy()
        live["date"] = pd.to_datetime(live["date"]).dt.normalize()
        live["fuel"] = live["fuel"].astype(str).str.upper()
        live_map = live.set_index(["date", "fuel"])["fair_price_eur_l"].to_dict()
        keys = list(zip(out["date"], out["fuel"]))
        live_values = pd.Series([live_map.get(k, np.nan) for k in keys], index=out.index)
        mask = live_values.notna()
        out.loc[mask, "fundamental_fair_price_eur_l"] = live_values[mask]
        out.loc[mask, "refined_quote_source"] = "LIVE_V0_5_1"
        out.loc[mask, "fundamental_fair_confidence"] = "HIGH"

    if live_market_snapshot is not None and not live_market_snapshot.empty:
        market = live_market_snapshot.copy()
        if "date" in market.columns:
            market["date"] = pd.to_datetime(market["date"]).dt.normalize()
        market["fuel"] = market["fuel"].astype(str).str.upper()
        if "refined_nowcast_eur_l" in market.columns:
            quote_map = market.set_index(["date", "fuel"])["refined_nowcast_eur_l"].to_dict()
            market_keys = list(zip(out["date"], out["fuel"]))
            market_values = pd.Series([quote_map.get(k, np.nan) for k in market_keys], index=out.index)
            market_mask = market_values.notna()
            out.loc[market_mask, "refined_quote_est_eur_l"] = market_values[market_mask]

    out["market_tension_cent_l"] = 100.0 * (
        out["observed_median_price_eur_l"] - out["fundamental_fair_price_eur_l"]
    )
    out["fundamental_fair_index"] = np.nan
    for fuel, idx in out.groupby("fuel").groups.items():
        vals = out.loc[idx, "fundamental_fair_price_eur_l"].astype(float)
        nonnull = vals.dropna()
        if nonnull.empty:
            continue
        base = float(nonnull.iloc[0])
        if base > 0:
            out.loc[idx, "fundamental_fair_index"] = 100.0 * vals / base

    return out.sort_values(["fuel", "date"]).reset_index(drop=True)
