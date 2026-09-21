from pathlib import Path

import pandas as pd

from fuel_fair_price.market.components import load_market_components
from fuel_fair_price.market.daily_proxy import (
    build_anchored_proxy_nowcast,
    extend_nowcast_daily,
    fit_brent_bridge_betas,
    latest_daily_market_snapshot,
    load_daily_product_proxies,
)
from fuel_fair_price.market.eu_history import load_france_weekly_htt
from fuel_fair_price.models.lag_selection import (
    best_filter_by_fuel,
    build_calibration_panel,
    score_filters,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_or_build_weekly() -> pd.DataFrame:
    path = ROOT / "config" / "eu_france_weekly_htt.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    df = load_france_weekly_htt(start="2020-01-01")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    weekly = _load_or_build_weekly()
    start = (weekly["date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    daily = load_daily_product_proxies(start=start)

    panel = build_calibration_panel(weekly, daily)
    scores = score_filters(panel, train_fraction=0.75, min_train=52)
    best = best_filter_by_fuel(scores)
    if not best:
        raise RuntimeError("No market timing filter could be selected")

    bridge_betas = fit_brent_bridge_betas(start="2023-01-01")

    model_rows = []
    for fuel, candidate in best.items():
        row = scores[(scores["fuel"] == fuel) & (scores["candidate"] == candidate)].iloc[0]
        model_rows.append(
            {
                "fuel": fuel,
                "selected_filter": candidate,
                "bridge_beta": bridge_betas.get(fuel, 1.0),
                "calibration_rmse_eur_l": row["rmse_eur_l"],
                "calibration_mae_eur_l": row["mae_eur_l"],
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
            }
        )

    model = pd.DataFrame(model_rows)
    model.to_csv(ROOT / "config" / "market_model.csv", index=False)

    anchors = load_market_components(ROOT / "config" / "market_components.csv")
    proxy_nowcast = build_anchored_proxy_nowcast(daily, anchors, best)
    target = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()
    extended = extend_nowcast_daily(
        proxy_nowcast,
        target_date=target,
        bridge_betas=bridge_betas,
    )
    snapshot = latest_daily_market_snapshot(extended, target_date=target)

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    scores.to_csv(output / "market_filter_scores.csv", index=False)
    extended.to_csv(output / "daily_refined_nowcast.csv", index=False)
    snapshot.to_csv(output / "live_market_snapshot.csv", index=False)

    print("\nFILTER SCORES")
    print(scores.to_string(index=False))
    print("\nMARKET MODEL")
    print(model.to_string(index=False))
    print("\nLIVE DAILY NOWCAST")
    cols = [
        "fuel", "date", "refined_nowcast_eur_l", "market_mode", "bridge_source",
        "effective_market_date", "product_proxy_date", "official_anchor_date",
        "market_freshness", "bridge_span_days", "official_anchor_age_days",
    ]
    # freshness dataclass uses 'status', rename only for human-readable output.
    view = snapshot.rename(columns={"status": "market_freshness"})
    print(view[[c for c in cols if c in view.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
