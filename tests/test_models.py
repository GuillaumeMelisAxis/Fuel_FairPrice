import math

import pandas as pd

from fuel_fair_price.models.anomaly import robust_zscore
from fuel_fair_price.models.brent import brent_usd_bbl_to_eur_l
from fuel_fair_price.models.fair_price import FairPriceInputs, compute_fair_price


def test_fair_price():
    p = compute_fair_price(FairPriceInputs(0.50, 0.20, 0.69, 0.20))
    assert math.isclose(p, 1.668, rel_tol=1e-12)


def test_brent_conversion():
    result = brent_usd_bbl_to_eur_l(80.0, 1.10)
    assert 0.45 < result < 0.47


def test_robust_zscore_center():
    z = robust_zscore(pd.Series([1.0, 1.0, 1.1, 1.2, 5.0]))
    assert z.iloc[-1] > 3
