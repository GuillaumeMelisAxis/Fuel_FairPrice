from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fuel_fair_price.analytics.index_series import (
    HistoricalIndexConfig,
    aggregate_live_anomaly_rate,
    build_chained_observed_index,
    build_historical_fundamental_fair_series,
    build_historical_local_anomaly_rate,
    build_historical_refined_quote_series,
    combine_index_series,
)
from fuel_fair_price.analytics.backtest import load_baseline_margins
from fuel_fair_price.market.components import load_market_components
from fuel_fair_price.market.daily_proxy import load_daily_product_proxies
from fuel_fair_price.market.eu_history import load_france_weekly_prices
from fuel_fair_price.models.historical_local import load_historical_model
from fuel_fair_price.visualization.station_map import build_station_map
from fuel_fair_price.visualization.index_charts import (
    plot_local_anomaly_rate,
    plot_market_tension,
    plot_observed_indices,
    plot_observed_and_fair_indices,
    plot_observed_vs_fair,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the v0.6.1 historical visual fuel index and station map."
    )
    parser.add_argument(
        "--panel",
        default=str(ROOT / "data" / "historical_station_panel.csv.gz"),
    )
    parser.add_argument(
        "--refresh-market-cache",
        action="store_true",
        help="Redownload EU weekly prices and FRED product proxies.",
    )
    parser.add_argument("--min-common-stations", type=int, default=30)
    return parser.parse_args()


def _load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/build_historical_panel.py first."
        )
    return pd.read_csv(
        path,
        parse_dates=["date", "updated_at"],
        dtype={
            "station_id": "string",
            "id": "string",
            "cp": "string",
            "code_departement": "string",
            "code_region": "string",
        },
    )


def _load_model():
    model_path = ROOT / "config" / "historical_local_model.csv"
    meta_path = ROOT / "config" / "historical_local_model_meta.json"
    persistent_path = ROOT / "config" / "station_persistent_bias.csv"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Historical model not found. Run scripts/train_historical_local_model.py first."
        )
    return load_historical_model(model_path, meta_path, persistent_path)


def _load_selected_filters() -> dict[str, str]:
    path = ROOT / "config" / "market_model.csv"
    df = pd.read_csv(path)
    return dict(
        zip(
            df["fuel"].astype(str).str.upper(),
            df["selected_filter"].astype(str),
        )
    )


def _load_fallback_excise() -> dict[str, float]:
    path = ROOT / "config" / "taxes.csv"
    df = pd.read_csv(path)
    df["fuel"] = df["fuel"].astype(str).str.upper()
    latest = df.sort_values("effective_from").groupby("fuel", as_index=False).tail(1)
    return dict(zip(latest["fuel"], latest["excise_eur_l"].astype(float)))


def _market_cache(panel_start: pd.Timestamp, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = ROOT / "data" / "index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    eu_path = cache_dir / "eu_france_weekly_prices.csv"
    proxy_path = cache_dir / "daily_product_proxies.csv.gz"

    if refresh or not eu_path.exists():
        print("Downloading EU Weekly Oil Bulletin history...")
        eu = load_france_weekly_prices(
            start=str((panel_start - pd.Timedelta(days=45)).date())
        )
        eu.to_csv(eu_path, index=False)
    else:
        eu = pd.read_csv(eu_path, parse_dates=["date"])

    if refresh or not proxy_path.exists():
        print("Downloading FRED refined-product proxies...")
        proxy = load_daily_product_proxies(
            start=str((panel_start - pd.Timedelta(days=45)).date())
        )
        proxy.to_csv(proxy_path, index=False, compression="gzip")
    else:
        proxy = pd.read_csv(
            proxy_path,
            parse_dates=["date", "spot_observed_date", "fx_observed_date"],
        )

    return eu, proxy


def _append_live_station_prices(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    live_scores_path = ROOT / "output" / "live_station_scores.csv"
    live_summary_path = ROOT / "output" / "live_index_summary.csv"
    if not live_scores_path.exists() or not live_summary_path.exists():
        return panel[["date", "fuel", "station_id", "price_eur_l"]].copy(), None, None

    live_summary = pd.read_csv(live_summary_path, parse_dates=["date"])
    live_scores = pd.read_csv(
        live_scores_path,
        dtype={"id": "string", "fuel": "string"},
    )
    if live_summary.empty or live_scores.empty:
        return panel[["date", "fuel", "station_id", "price_eur_l"]].copy(), None, None

    index_date = pd.Timestamp(live_summary["date"].max()).normalize()
    live_obs = live_scores[["id", "fuel", "price_eur_l"]].copy()
    live_obs = live_obs.rename(columns={"id": "station_id"})
    live_obs["date"] = index_date
    live_obs["station_id"] = live_obs["station_id"].astype("string")

    station_history = pd.concat(
        [
            panel[["date", "fuel", "station_id", "price_eur_l"]].copy(),
            live_obs[["date", "fuel", "station_id", "price_eur_l"]],
        ],
        ignore_index=True,
    )
    return station_history, live_scores, live_summary


def _load_live_market_snapshot() -> pd.DataFrame | None:
    path = ROOT / "output" / "live_market_snapshot.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["fuel"] = frame["fuel"].astype(str).str.upper()
    return frame


def _write_dashboard(index_series: pd.DataFrame, output_dir: Path, *, map_available: bool) -> Path:
    latest = (
        index_series.sort_values("date")
        .groupby("fuel", as_index=False)
        .tail(1)
        .sort_values("fuel")
    )

    cards = []
    for _, r in latest.iterrows():
        cards.append(
            f"""<div class="metric-card">
              <div class="fuel">{r['fuel']}</div>
              <div class="price">{r['observed_median_price_eur_l']:.3f} €/L</div>
              <div class="metric"><span>Fundamental fair</span><b>{r['fundamental_fair_price_eur_l']:.3f} €/L</b></div>
              <div class="metric"><span>Market tension</span><b>{r['market_tension_cent_l']:+.1f} c/L</b></div>
              <div class="metric"><span>Local anomaly rate</span><b>{r['local_anomaly_rate_pct']:.1f}%</b></div>
              <div class="metric"><span>Observed index</span><b>{r['observed_price_index']:.1f}</b></div>
            </div>"""
        )

    map_section = ""
    if map_available:
        map_section = """
        <section>
          <h2>Station-level local price excess</h2>
          <p class="section-sub">Interactive station map. Green stations are cheap relative to their local fair benchmark; red stations are expensive relative to comparable local peers. Use the layer control to switch fuel. Grey points are stale, low-confidence or unassessed.</p>
          <div class="map-card"><iframe src="station_map.html" title="Fuel station anomaly map"></iframe></div>
          <p class="disclaimer">The map is an economic anomaly indicator, not a legal finding of abusive pricing.</p>
        </section>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fuel Fair Price Index — France</title>
<style>
:root {{ --text:#202124; --muted:#667085; --border:#e4e7ec; --bg:#f8fafc; }}
* {{ box-sizing:border-box; }}
body {{ font-family:Arial,sans-serif; max-width:1280px; margin:0 auto; padding:32px 22px 54px; color:var(--text); background:white; }}
h1 {{ margin:0 0 6px; font-size:2rem; }}
h2 {{ margin:36px 0 6px; }}
.sub,.section-sub,.note,.disclaimer {{ color:var(--muted); line-height:1.5; }}
.summary {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin:24px 0 30px; }}
.metric-card {{ border:1px solid var(--border); border-radius:14px; padding:18px; background:#fff; box-shadow:0 1px 3px rgba(16,24,40,.04); }}
.fuel {{ font-size:.9rem; color:var(--muted); font-weight:700; letter-spacing:.08em; }}
.price {{ font-size:2rem; font-weight:700; margin:5px 0 15px; }}
.metric {{ display:flex; justify-content:space-between; gap:14px; padding:5px 0; border-top:1px solid #f0f1f3; font-size:.94rem; }}
.metric span {{ color:var(--muted); }}
.hero-card,.card,.map-card {{ border:1px solid var(--border); border-radius:14px; padding:12px; background:#fff; box-shadow:0 1px 3px rgba(16,24,40,.04); }}
.hero-card img,.card img {{ width:100%; height:auto; display:block; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:18px; }}
.map-card {{ padding:0; overflow:hidden; margin-top:14px; }}
.map-card iframe {{ width:100%; height:720px; border:0; display:block; }}
.note {{ margin-top:30px; font-size:.9rem; border-top:1px solid var(--border); padding-top:18px; }}
.disclaimer {{ font-size:.85rem; }}
@media (max-width:820px) {{ .summary,.grid {{ grid-template-columns:1fr; }} .map-card iframe {{ height:620px; }} }}
</style>
</head>
<body>
<h1>Fuel Fair Price Index — France</h1>
<div class="sub">v0.6.1 dashboard. Observed indices are model-free; fair-value and local-anomaly series are explicitly model/version dependent.</div>
<div class="summary">{''.join(cards)}</div>

<section>
  <h2>Observed vs fundamental indices</h2>
  <p class="section-sub">SP95 and Gazole on one base-100 chart. Solid lines are observed chained-station indices; dashed lines are fundamental fair-value indices.</p>
  <div class="hero-card"><img src="fuel_observed_vs_fair_base100.png" alt="SP95 and Gazole observed versus fundamental fair indices"></div>
</section>

{map_section}

<section>
  <h2>Detailed views</h2>
  <div class="grid">
    <div class="card"><img src="sp95_observed_vs_fair.png" alt="SP95 observed versus fair"></div>
    <div class="card"><img src="gazole_observed_vs_fair.png" alt="Gazole observed versus fair"></div>
    <div class="card"><img src="market_tension.png" alt="Market tension"></div>
    <div class="card"><img src="local_anomaly_rate.png" alt="Local anomaly rate"></div>
    <div class="card"><img src="fuel_indices_base100.png" alt="Observed fuel indices base 100"></div>
  </div>
</section>

<div class="note">
<b>Method notes.</b> Observed price index: chained median station-to-station price relatives using stations present on consecutive dates.
Fundamental fair: DGEC refined quotations when available; otherwise DGEC-anchored public product-proxy reconstruction. Historical tax wedge comes from EU Weekly Oil Bulletin France TTC/HTT data.
Local anomaly rate: retrospective application of the frozen v0.5.1 local model to each historical station snapshot.
</div>
</body></html>"""
    path = output_dir / "index_dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path

def main() -> None:
    args = _parse_args()
    panel = _load_panel(Path(args.panel))
    model, meta, _ = _load_model()

    station_history, live_scores, live_summary = _append_live_station_prices(panel)
    live_market_snapshot = _load_live_market_snapshot()
    index_cfg = HistoricalIndexConfig(min_common_stations=int(args.min_common_stations))

    print("Building composition-resistant observed indices...")
    observed = build_chained_observed_index(station_history, config=index_cfg)

    print("Reconstructing historical local anomaly rate...")
    anomaly = build_historical_local_anomaly_rate(panel, model=model, meta=meta)
    if live_scores is not None and live_summary is not None:
        live_date = pd.Timestamp(live_summary["date"].max())
        anomaly = pd.concat(
            [anomaly, aggregate_live_anomaly_rate(live_scores, live_date)],
            ignore_index=True,
        ).drop_duplicates(["date", "fuel"], keep="last")

    panel_start = pd.Timestamp(panel["date"].min())
    eu_prices, daily_proxy = _market_cache(panel_start, bool(args.refresh_market_cache))

    anchors = load_market_components(ROOT / "config" / "market_components.csv")
    selected_filters = _load_selected_filters()
    dates_by_fuel = observed[["date", "fuel"]].drop_duplicates()

    print("Reconstructing historical fundamental fair price...")
    refined = build_historical_refined_quote_series(
        dates_by_fuel,
        daily_proxy=daily_proxy,
        anchors=anchors,
        selected_filter=selected_filters,
    )
    baseline = load_baseline_margins(str(ROOT / "config" / "baseline_margins.csv"))
    fair = build_historical_fundamental_fair_series(
        dates_by_fuel,
        weekly_france_prices=eu_prices,
        refined_series=refined,
        baseline_margins=baseline,
        fallback_excise=_load_fallback_excise(),
        config=index_cfg,
    )

    series = combine_index_series(
        observed,
        fair,
        anomaly,
        live_summary=live_summary,
        live_market_snapshot=live_market_snapshot,
    )

    output_dir = ROOT / "output" / "index"
    output_dir.mkdir(parents=True, exist_ok=True)
    series_path = output_dir / "historical_fuel_index.csv"
    series.to_csv(series_path, index=False)

    plot_observed_vs_fair(series, "SP95", output_dir / "sp95_observed_vs_fair.png")
    plot_observed_vs_fair(series, "GAZOLE", output_dir / "gazole_observed_vs_fair.png")
    plot_observed_indices(series, output_dir / "fuel_indices_base100.png")
    plot_observed_and_fair_indices(series, output_dir / "fuel_observed_vs_fair_base100.png")
    plot_market_tension(series, output_dir / "market_tension.png")
    plot_local_anomaly_rate(series, output_dir / "local_anomaly_rate.png")
    map_available = live_scores is not None and not live_scores.empty
    if map_available:
        print("Building interactive station map...")
        build_station_map(live_scores, output_dir / "station_map.html")
    dashboard = _write_dashboard(series, output_dir, map_available=map_available)

    metadata = {
        "version": "0.6.1",
        "observed_index_method": "CHAINED_MEDIAN_MATCHED_STATION_RELATIVES",
        "base_value": float(index_cfg.base_value),
        "local_model_version": "v0.5.1_final",
        "local_anomaly_series": "RETROSPECTIVE_MODEL_DEPENDENT",
        "fundamental_fair_method": "DGEC_OFFICIAL_OR_DGEC_ANCHORED_PRODUCT_PROXY",
        "historical_tax_method": "EU_WEEKLY_TTC_HTT_WEDGE",
        "first_date": str(pd.Timestamp(series["date"].min()).date()),
        "last_date": str(pd.Timestamp(series["date"].max()).date()),
        "station_map_metric": "local_excess_cent_l",
        "station_map_interpretation": "ECONOMIC_ANOMALY_NOT_LEGAL_FINDING",
    }
    (output_dir / "index_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nINDEX BUILT")
    print(series_path)
    print(dashboard)
    cols = [
        "date", "fuel", "observed_median_price_eur_l",
        "fundamental_fair_price_eur_l", "market_tension_cent_l",
        "local_anomaly_rate_pct", "observed_price_index",
        "fundamental_fair_confidence",
    ]
    print("\nLatest values")
    print(series.sort_values("date").groupby("fuel", as_index=False).tail(1)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
