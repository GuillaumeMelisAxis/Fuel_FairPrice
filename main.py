from pathlib import Path

from fuel_fair_price.data.stations import clean_stations, fetch_current_stations, to_long_format
from fuel_fair_price.data.taxes import get_tax_row, load_taxes
from fuel_fair_price.market.components import add_normal_distribution_margin, load_market_components
from fuel_fair_price.models.anomaly import score_stations
from fuel_fair_price.models.fair_price import FairPriceInputs, compute_fair_price

ROOT = Path(__file__).resolve().parent


def main() -> None:
    market = load_market_components(ROOT / "config" / "market_components.csv")
    if market.empty:
        raise RuntimeError(
            "config/market_components.csv is empty. Add weekly DGEC refined quotes and "
            "observed transport-distribution margins before running the index."
        )

    market = add_normal_distribution_margin(market)
    taxes = load_taxes(ROOT / "config" / "taxes.csv")

    raw_stations = fetch_current_stations()
    stations = to_long_format(clean_stations(raw_stations))

    outputs = []
    for fuel in ["SP95", "GAZOLE"]:
        m = market[market["fuel"] == fuel].dropna(subset=["normal_distribution_margin_eur_l"]).iloc[-1]
        tax = get_tax_row(taxes, fuel=fuel, as_of=m["date"])

        fair = compute_fair_price(
            FairPriceInputs(
                refined_quote_eur_l=float(m["refined_quote_eur_l"]),
                normal_distribution_margin_eur_l=float(m["normal_distribution_margin_eur_l"]),
                excise_eur_l=float(tax["excise_eur_l"]),
                vat_rate=float(tax["vat_rate"]),
            )
        )

        subset = stations[stations["fuel"] == fuel]
        scored = score_stations(subset, fair_price_eur_l=fair)
        outputs.append(scored)

        print(f"{fuel}: fair price = {fair:.3f} EUR/L; stations = {len(scored):,}")
        print(scored[["id", "ville", "price_eur_l", "fair_price_eur_l", "spread_cent_l", "anomaly_z", "flag"]].head(10))
        print()


if __name__ == "__main__":
    main()
