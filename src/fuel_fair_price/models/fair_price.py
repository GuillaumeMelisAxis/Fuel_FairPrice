from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FairPriceInputs:
    refined_quote_eur_l: float
    normal_distribution_margin_eur_l: float
    excise_eur_l: float
    vat_rate: float


def compute_fair_price(x: FairPriceInputs) -> float:
    """
    Fair pump price in EUR/L.

    refined quotation + normal transport/distribution + excise are summed before
    VAT. The refined quotation already embeds crude-oil/refining market economics,
    so Brent must not be added again to this formula.
    """
    pre_vat = (
        x.refined_quote_eur_l
        + x.normal_distribution_margin_eur_l
        + x.excise_eur_l
    )
    return pre_vat * (1.0 + x.vat_rate)


def implied_distribution_margin(
    observed_price_eur_l: float,
    refined_quote_eur_l: float,
    excise_eur_l: float,
    vat_rate: float,
) -> float:
    """Infer the station's gross transport/distribution component in EUR/L."""
    return observed_price_eur_l / (1.0 + vat_rate) - excise_eur_l - refined_quote_eur_l
