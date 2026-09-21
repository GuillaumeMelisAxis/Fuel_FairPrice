from __future__ import annotations

import pandas as pd

from fuel_fair_price.models.anomaly import robust_zscore


def score_historical_station_snapshot(
    snapshot: pd.DataFrame,
    fair_by_month: pd.DataFrame,
    peer_columns: tuple[str, ...] = ("date", "fuel", "road_type"),
) -> pd.DataFrame:
    """Attach monthly fair prices and score station residuals against peers."""
    fair = fair_by_month[["date", "fuel", "fair_price_eur_l"]].copy()
    fair["date"] = pd.to_datetime(fair["date"])
    out = snapshot.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["fuel"] = out["fuel"].str.upper()
    out = out.merge(fair, on=["date", "fuel"], how="inner")
    out["spread_eur_l"] = out["price_eur_l"] - out["fair_price_eur_l"]
    out["spread_cent_l"] = 100.0 * out["spread_eur_l"]
    valid = [c for c in peer_columns if c in out.columns]
    if valid:
        out["anomaly_z"] = out.groupby(valid, dropna=False)["spread_eur_l"].transform(robust_zscore)
    else:
        out["anomaly_z"] = robust_zscore(out["spread_eur_l"])
    return out.sort_values(["date", "fuel", "anomaly_z"], ascending=[True, True, False])
