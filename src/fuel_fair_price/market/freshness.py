from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketFreshness:
    target_date: pd.Timestamp
    effective_market_date: pd.Timestamp
    product_proxy_date: pd.Timestamp
    official_anchor_date: pd.Timestamp
    market_mode: str
    bridge_source: str
    market_age_days: int
    market_business_lag: int
    product_proxy_age_days: int
    official_anchor_age_days: int
    bridge_span_days: int
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _date(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Europe/Paris").tz_localize(None)
    return ts.normalize()


def business_lag_days(observation_date, target_date) -> int:
    """Approximate number of business days between two market dates."""
    obs = _date(observation_date)
    target = _date(target_date)
    if target <= obs:
        return 0
    return int(np.busday_count(obs.date(), target.date()))


def assess_market_freshness(
    *,
    target_date,
    effective_market_date,
    product_proxy_date,
    official_anchor_date,
    market_mode: str,
    bridge_source: str,
) -> MarketFreshness:
    """Classify the reliability of the live market component.

    CURRENT
        Market observation no more than one business day behind, the Brent/FX
        bridge spans at most seven calendar days, and the official DGEC anchor
        is no more than 21 days old.

    STALE
        Usable for indicative pricing, but at least one of those limits is
        exceeded.  We allow up to three business days, a 14-day bridge and a
        35-day official anchor.

    VERY_STALE
        The market component is too old for an actionable station flag.  The
        fair price can still be displayed for diagnostics.
    """
    target = _date(target_date)
    effective = _date(effective_market_date)
    proxy = _date(product_proxy_date)
    anchor = _date(official_anchor_date)

    market_age = max(0, int((target - effective).days))
    proxy_age = max(0, int((target - proxy).days))
    anchor_age = max(0, int((target - anchor).days))
    bridge_span = max(0, int((target - proxy).days)) if market_mode == "BRENT_FX_BRIDGE" else 0
    business_lag = business_lag_days(effective, target)

    if business_lag <= 1 and bridge_span <= 7 and anchor_age <= 21:
        status = "CURRENT"
    elif business_lag <= 3 and bridge_span <= 14 and anchor_age <= 35:
        status = "STALE"
    else:
        status = "VERY_STALE"

    return MarketFreshness(
        target_date=target,
        effective_market_date=effective,
        product_proxy_date=proxy,
        official_anchor_date=anchor,
        market_mode=market_mode,
        bridge_source=bridge_source,
        market_age_days=market_age,
        market_business_lag=business_lag,
        product_proxy_age_days=proxy_age,
        official_anchor_age_days=anchor_age,
        bridge_span_days=bridge_span,
        status=status,
    )
