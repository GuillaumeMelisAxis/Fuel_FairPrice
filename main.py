from pathlib import Path

import pandas as pd

from fuel_fair_price.data.stations import clean_stations, fetch_current_stations, to_long_format
from fuel_fair_price.data.taxes import get_tax_row, load_taxes
from fuel_fair_price.market.live import build_live_market_snapshot, load_baseline_margins
from fuel_fair_price.models.anomaly import score_stations
from fuel_fair_price.models.fair_price import FairPriceInputs, compute_fair_price

ROOT = Path(__file__).resolve().parent
MAX_STATION_AGE_DAYS = 30.0


def main() -> None:
    target = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()

    baseline_by_fuel = load_baseline_margins(ROOT / "config" / "baseline_margins.csv")
    taxes = load_taxes(ROOT / "config" / "taxes.csv")

    daily_nowcast, market_snapshot = build_live_market_snapshot(ROOT, target_date=target)
    if market_snapshot.empty:
        raise RuntimeError("Could not build the live market snapshot")

    # Standardise the freshness column name used in downstream output.
    market_snapshot = market_snapshot.rename(columns={"status": "market_freshness"})

    raw_stations = fetch_current_stations()
    stations = to_long_format(clean_stations(raw_stations))
    stations = stations[(stations["age_hours"] >= 0)].copy()

    output_rows = []
    scored_frames = []

    print(f"INDEX DATE: {target.date()}")
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
        scoring_sample = all_fuel[all_fuel["price_age_days"] <= MAX_STATION_AGE_DAYS].copy()

        freshness = str(m["market_freshness"])
        scored = score_stations(
            scoring_sample,
            fair_price_eur_l=fair,
            market_freshness=freshness,
        )
        scored["market_mode"] = m["market_mode"]
        scored["market_data_date"] = m["effective_market_date"]
        scored["official_anchor_date"] = m["official_anchor_date"]
        scored_frames.append(scored)

        median_price = float(scored["price_eur_l"].median()) if not scored.empty else float("nan")
        median_spread = float(scored["spread_cent_l"].median()) if not scored.empty else float("nan")

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
            f"fair price = {fair:.3f} EUR/L | median station = {median_price:.3f} EUR/L | "
            f"median spread = {median_spread:+.2f} c/L"
        )
        print(
            f"stations raw={len(all_fuel):,} | scored (<= {MAX_STATION_AGE_DAYS:.0f}d)={len(scored):,}"
        )
        if freshness == "VERY_STALE":
            print("WARNING: market data are VERY_STALE; anomaly flags are disabled.")

        columns = [
            "id", "ville", "price_eur_l", "updated_at", "price_age_days",
            "fair_price_eur_l", "spread_cent_l", "anomaly_z", "flag",
        ]
        print(scored[columns].head(10).to_string(index=False))
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
                "stations_scored": len(scored),
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
    pd.DataFrame(output_rows).to_csv(output_dir / "live_index_summary.csv", index=False)
    market_snapshot.to_csv(output_dir / "live_market_snapshot.csv", index=False)
    if scored_frames:
        pd.concat(scored_frames, ignore_index=True).to_csv(output_dir / "live_station_scores.csv", index=False)
    daily_nowcast.to_csv(output_dir / "daily_refined_nowcast.csv", index=False)


if __name__ == "__main__":
    main()
