from datetime import date

from fuel_fair_price.market.dgec import parse_npg_text


def test_parse_npg_text_pair():
    text = """
    Cotations internationales et prix hors toutes taxes
    Supercarburant sans plomb (sp95-e5) en c€/l
    Moyenne mensuelle févr.-26 janv.-26 variation
    France 75,66 73,74 1,92
    U.E. 27 70,97 69,22 1,75
    Cotations 44,38 41,61 2,77
    Gazole en c€/l
    Moyenne mensuelle févr.-26 janv.-26 variation
    France 78,62 75,45 3,18
    U.E. 27 78,21 75,95 2,27
    Cotations 50,37 46,95 3,43
    Fioul domestique
    """
    df = parse_npg_text(text, date(2026, 2, 27), "example")
    assert len(df) == 4
    sp = df[(df.fuel == "SP95") & (df.date.dt.month == 2)].iloc[0]
    assert abs(sp.refined_quote_eur_l - 0.4438) < 1e-12
    dz = df[(df.fuel == "GAZOLE") & (df.date.dt.month == 1)].iloc[0]
    assert abs(dz.france_htt_eur_l - 0.7545) < 1e-12
