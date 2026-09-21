BARREL_LITERS = 158.987294928


def brent_usd_bbl_to_eur_l(brent_usd_bbl: float, eurusd_usd_per_eur: float) -> float:
    """Convert Brent USD/bbl to EUR/L using EURUSD quoted as USD per EUR."""
    if eurusd_usd_per_eur <= 0:
        raise ValueError("EURUSD must be strictly positive")
    return brent_usd_bbl / eurusd_usd_per_eur / BARREL_LITERS


def refined_vs_crude_spread(
    refined_quote_eur_l: float,
    brent_usd_bbl: float,
    eurusd_usd_per_eur: float,
) -> float:
    """Diagnostic spread between refined-product quotation and crude equivalent."""
    crude_eur_l = brent_usd_bbl_to_eur_l(brent_usd_bbl, eurusd_usd_per_eur)
    return refined_quote_eur_l - crude_eur_l
