from __future__ import annotations

import numpy as np
import pandas as pd

MAD_NORMALIZER = 1.4826


def robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - median) / (MAD_NORMALIZER * mad)


def score_stations(
    stations: pd.DataFrame,
    fair_price_eur_l: float,
    peer_columns: tuple[str, ...] = ("region", "road_type", "fuel"),
    market_freshness: str = "CURRENT",
) -> pd.DataFrame:
    """Compute fundamental spread + peer-relative robust anomaly score."""
    out = stations.copy()
    out["fair_price_eur_l"] = fair_price_eur_l
    out["spread_eur_l"] = out["price_eur_l"] - fair_price_eur_l
    out["spread_cent_l"] = 100.0 * out["spread_eur_l"]

    valid_peer_columns = [c for c in peer_columns if c in out.columns]
    if valid_peer_columns:
        out["anomaly_z"] = out.groupby(valid_peer_columns, dropna=False)["spread_eur_l"].transform(robust_zscore)
    else:
        out["anomaly_z"] = robust_zscore(out["spread_eur_l"])

    out["flag"] = pd.cut(
        out["anomaly_z"],
        bins=[-np.inf, 2.0, 3.0, np.inf],
        labels=["NORMAL", "HIGH", "VERY_HIGH"],
        right=True,
    ).astype("string")
    out["market_freshness"] = market_freshness
    out["actionable"] = market_freshness != "VERY_STALE"
    if market_freshness == "VERY_STALE":
        out["flag"] = "MARKET_DATA_STALE"
    return out.sort_values(["anomaly_z", "spread_eur_l"], ascending=False)
