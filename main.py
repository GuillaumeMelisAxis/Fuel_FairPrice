from pathlib import Path

import pandas as pd

from fuel_fair_price.data.stations import clean_stations, fetch_current_stations, to_long_format
from fuel_fair_price.data.taxes import get_tax_row, load_taxes
from fuel_fair_price.market.live import build_live_market_snapshot, load_baseline_margins
from fuel_fair_price.models.anomaly import score_stations
from fuel_fair_price.models.fair_price import FairPriceInputs, compute_fair_price
from fuel_fair_price.models.historical_local import (
    apply_historical_local_model,
    load_historical_model,
)
from fuel_fair_price.models.local_adjustment import (
    add_competition_features,
    apply_local_adjustment,
)
from fuel_fair_price.models.logistics import load_logistics_overrides

ROOT = Path(__file__).resolve().parent
MAX_STATION_AGE_DAYS = 30.0

HIST_MODEL_PATH = ROOT / "config" / "historical_local_model.csv"
HIST_META_PATH = ROOT / "config" / "historical_local_model_meta.json"
PERSISTENT_BIAS_PATH = ROOT / "config" / "station_persistent_bias.csv"


def _load_v051_model():
    if not (HIST_MODEL_PATH.exists() and HIST_META_PATH.exists()):
        return None
    return load_historical_model(
        HIST_MODEL_PATH,
        HIST_META_PATH,
        PERSISTENT_BIAS_PATH,
    )


def main() -> None:
    target = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()

    baseline_by_fuel = load_baseline_margins(ROOT / "config" / "baseline_margins.csv")
    taxes = load_taxes(ROOT / "config" / "taxes.csv")
    logistics_overrides = load_logistics_overrides(
        ROOT / "config" / "logistics_overrides.csv"
    )
    historical_bundle = _load_v051_model()

    daily_nowcast, market_snapshot = build_live_market_snapshot(ROOT, target_date=target)
    if market_snapshot.empty:
        raise RuntimeError("Could not build the live market snapshot")
    market_snapshot = market_snapshot.rename(columns={"status": "market_freshness"})

    raw_stations = fetch_current_stations()
    stations = to_long_format(clean_stations(raw_stations))
    stations = stations[(stations["age_hours"] >= 0)].copy()

    output_rows = []
    scored_frames = []
    local_effect_frames = []

    print(f"INDEX DATE: {target.date()}")
    if historical_bundle is None:
        print(
            "LOCAL MODEL: CROSS_SECTIONAL_FALLBACK (v0.4.2 logic)\n"
            "Run `python scripts/build_historical_panel.py` then "
            "`python scripts/train_historical_local_model.py` to activate v0.5.1."
        )
    else:
        _, historical_meta, _ = historical_bundle
        print("LOCAL MODEL: HISTORICAL_PANEL v0.5.1-final")
        fuel_ranges = []
        for fuel, info in historical_meta.get("fuels", {}).items():
            fuel_ranges.append(
                f"{fuel}={info['train_start']}->{info['train_end']} "
                f"({info['n_months']} months)"
            )
        if fuel_ranges:
            print("TRAINING WINDOW: " + " | ".join(fuel_ranges))
    print()

    for fuel in ["SP95", "GAZOLE"]:
        market = market_snapshot[market_snapshot["fuel"] == fuel]
        if market.empty:
            raise RuntimeError(f"No live market row for {fuel}")
        m = market.iloc[-1]

        tax = get_tax_row(taxes, fuel=fuel, as_of=target)
        baseline_margin = float(baseline_by_fuel[fuel])
        refined = float(m["refined_nowcast_eur_l"])

        fair = compute_fair_price(
            FairPriceInputs(
                refined_quote_eur_l=refined,
                normal_distribution_margin_eur_l=baseline_margin,
                excise_eur_l=float(tax["excise_eur_l"]),
                vat_rate=float(tax["vat_rate"]),
            )
        )

        all_fuel = stations[stations["fuel"] == fuel].copy()
        # Physical/local-market feature: compute it on the full current fuel
        # universe before removing stale price observations.
        all_fuel = add_competition_features(all_fuel)
        scoring_sample = all_fuel[
            all_fuel["price_age_days"] <= MAX_STATION_AGE_DAYS
        ].copy()

        freshness = str(m["market_freshness"])
        scored = score_stations(
            scoring_sample,
            fair_price_eur_l=fair,
            market_freshness=freshness,
        )
        scored["market_mode"] = m["market_mode"]
        scored["market_data_date"] = m["effective_market_date"]
        scored["official_anchor_date"] = m["official_anchor_date"]

        if historical_bundle is not None:
            model, meta, persistent = historical_bundle
            locally_scored = apply_historical_local_model(
                scored,
                model=model,
                meta=meta,
                logistics_overrides=logistics_overrides,
                persistent_bias=persistent,
            )
            local_model_type = "HISTORICAL_PANEL"
        else:
            locally_scored, local_effects = apply_local_adjustment(
                scored,
                logistics_overrides=logistics_overrides,
            )
            locally_scored["local_model_type"] = "CROSS_SECTIONAL_FALLBACK"
            local_model_type = "CROSS_SECTIONAL_FALLBACK"
            if not local_effects.empty:
                local_effect_frames.append(local_effects)

        scored_frames.append(locally_scored)

        median_price = (
            float(locally_scored["price_eur_l"].median())
            if not locally_scored.empty
            else float("nan")
        )
        median_spread = (
            float(locally_scored["spread_cent_l"].median())
            if not locally_scored.empty
            else float("nan")
        )
        median_local_premium = (
            float(locally_scored["structural_local_premium_cent_l"].median())
            if not locally_scored.empty
            else float("nan")
        )
        median_local_residual = (
            float(locally_scored["local_residual_cent_l"].median())
            if not locally_scored.empty
            else float("nan")
        )

        print(f"=== {fuel} ===")
        print(
            f"market quote = {refined:.4f} EUR/L | mode={m['market_mode']} | "
            f"source={m['bridge_source']} | freshness={freshness}"
        )
        print(
            f"market data={pd.Timestamp(m['effective_market_date']).date()} | "
            f"product proxy={pd.Timestamp(m['product_proxy_date']).date()} | "
            f"DGEC anchor={pd.Timestamp(m['official_anchor_date']).date()} | "
            f"bridge span={int(m['bridge_span_days'])}d"
        )
        print(
            f"fair price = {fair:.3f} EUR/L | median station = "
            f"{median_price:.3f} EUR/L | median spread = {median_spread:+.2f} c/L"
        )
        print(
            f"stations raw={len(all_fuel):,} | "
            f"scored (<= {MAX_STATION_AGE_DAYS:.0f}d)={len(locally_scored):,}"
        )
        print(
            f"local model={local_model_type} | "
            f"median structural premium={median_local_premium:+.2f} c/L | "
            f"median local residual={median_local_residual:+.2f} c/L"
        )
        if "structural_premium_confidence" in locally_scored.columns:
            print(
                "structural premium confidence: "
                f"{locally_scored['structural_premium_confidence'].value_counts().to_dict()}"
            )
        if "peer_confidence" in locally_scored.columns:
            print(
                f"peer confidence: "
                f"{locally_scored['peer_confidence'].value_counts().to_dict()}"
            )
        if "accessibility_class" in locally_scored.columns:
            print(
                f"accessibility classes: "
                f"{locally_scored['accessibility_class'].value_counts().to_dict()}"
            )
        if freshness == "VERY_STALE":
            print("WARNING: market data are VERY_STALE; anomaly flags are disabled.")

        columns = [
            "id",
            "ville",
            "road_type",
            "price_eur_l",
            "updated_at",
            "price_age_days",
            "fair_price_eur_l",
            "spread_cent_l",
            "accessibility_class",
            "logistics_peer_group",
            "base_structural_local_premium_cent_l",
            "logistics_premium_cent_l",
            "logistics_confidence",
            "logistics_status",
            "structural_local_premium_cent_l",
            "structural_premium_confidence",
            "historical_model_coverage",
            "persistent_station_bias_cent_l",
            "persistent_bias_months",
            "persistent_bias_confidence",
            "persistence_status",
            "local_fair_price_eur_l",
            "local_residual_cent_l",
            "nearest_station_km",
            "stations_within_10km",
            "local_peer_radius_km",
            "local_peer_count",
            "local_peer_median_residual_cent_l",
            "local_excess_cent_l",
            "local_peer_scale_cent_l",
            "local_anomaly_score",
            "peer_confidence",
            "peer_selection_method",
            "local_flag",
        ]
        columns = [c for c in columns if c in locally_scored.columns]
        print(locally_scored[columns].head(10).to_string(index=False))
        print()

        output_rows.append(
            {
                "date": target,
                "fuel": fuel,
                "refined_nowcast_eur_l": refined,
                "baseline_margin_eur_l": baseline_margin,
                "fair_price_eur_l": fair,
                "median_station_price_eur_l": median_price,
                "median_spread_cent_l": median_spread,
                "stations_scored": len(locally_scored),
                "median_structural_local_premium_cent_l": median_local_premium,
                "median_local_residual_cent_l": median_local_residual,
                "local_model_type": local_model_type,
                "market_mode": m["market_mode"],
                "market_freshness": freshness,
                "market_data_date": m["effective_market_date"],
                "product_proxy_date": m["product_proxy_date"],
                "official_anchor_date": m["official_anchor_date"],
                "bridge_span_days": m["bridge_span_days"],
                "official_anchor_age_days": m["official_anchor_age_days"],
            }
        )

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame(output_rows).to_csv(
        output_dir / "live_index_summary.csv",
        index=False,
    )
    market_snapshot.to_csv(
        output_dir / "live_market_snapshot.csv",
        index=False,
    )
    if scored_frames:
        all_scored = pd.concat(scored_frames, ignore_index=True)
        all_scored.to_csv(
            output_dir / "live_station_scores.csv",
            index=False,
        )
        logistics_aware = all_scored[
            all_scored.get("accessibility_class", "MAINLAND") != "MAINLAND"
        ].copy()
        if not logistics_aware.empty:
            logistics_aware.to_csv(
                output_dir / "logistics_aware_stations.csv",
                index=False,
            )
    if local_effect_frames:
        pd.concat(local_effect_frames, ignore_index=True).to_csv(
            output_dir / "local_adjustment_effects.csv",
            index=False,
        )
    daily_nowcast.to_csv(
        output_dir / "daily_refined_nowcast.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
