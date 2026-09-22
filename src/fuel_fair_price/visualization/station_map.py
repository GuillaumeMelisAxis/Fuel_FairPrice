from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


_COLOR_BINS = (
    (-np.inf, -5.0, "#1a9850", "≤ -5 c/L"),
    (-5.0, 0.0, "#91cf60", "-5 to 0 c/L"),
    (0.0, 5.0, "#ffffbf", "0 to +5 c/L"),
    (5.0, 10.0, "#fdae61", "+5 to +10 c/L"),
    (10.0, 20.0, "#f46d43", "+10 to +20 c/L"),
    (20.0, np.inf, "#d73027", "> +20 c/L"),
)


def local_excess_color(excess_cent_l: float | int | None) -> str:
    """Map a local excess in c/L to the dashboard green-to-red scale."""
    try:
        value = float(excess_cent_l)
    except (TypeError, ValueError):
        return "#9e9e9e"
    if not np.isfinite(value):
        return "#9e9e9e"
    for lower, upper, color, _ in _COLOR_BINS:
        if lower < value <= upper:
            return color
    return "#9e9e9e"


def _coordinates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if {"latitude_deg", "longitude_deg"}.issubset(frame.columns):
        lat = pd.to_numeric(frame["latitude_deg"], errors="coerce")
        lon = pd.to_numeric(frame["longitude_deg"], errors="coerce")
        return lat, lon

    lat = pd.to_numeric(frame.get("latitude"), errors="coerce")
    lon = pd.to_numeric(frame.get("longitude"), errors="coerce")
    # Official station feed frequently stores WGS84 coordinates x100000.
    lat = lat.where(lat.abs() <= 90, lat / 100000.0)
    lon = lon.where(lon.abs() <= 180, lon / 100000.0)
    return lat, lon


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}{suffix}"


def _popup(row: pd.Series) -> str:
    ville = escape(str(row.get("ville", "")))
    address = escape(str(row.get("adresse", "")))
    fuel = escape(str(row.get("fuel", "")))
    station_id = escape(str(row.get("id", row.get("station_id", ""))))
    local_flag = escape(str(row.get("local_flag", "")))
    peer_conf = escape(str(row.get("peer_confidence", "")))
    persistence = escape(str(row.get("persistence_status", "n/a")))
    updated = escape(str(row.get("updated_at", "n/a")))
    return f"""
    <div style="font-family:Arial,sans-serif;min-width:250px;line-height:1.45">
      <b>{ville}</b><br>
      <span style="color:#666">{address}</span><br>
      <b>Fuel:</b> {fuel}<br>
      <b>Station ID:</b> {station_id}<br>
      <hr style="border:none;border-top:1px solid #ddd">
      <b>Observed:</b> {_fmt(row.get('price_eur_l'), 3, ' €/L')}<br>
      <b>Local fair:</b> {_fmt(row.get('local_fair_price_eur_l'), 3, ' €/L')}<br>
      <b>Local excess:</b> {_fmt(row.get('local_excess_cent_l'), 1, ' c/L')}<br>
      <b>Anomaly score:</b> {_fmt(row.get('local_anomaly_score'), 2)}<br>
      <b>Flag:</b> {local_flag}<br>
      <b>Peer confidence:</b> {peer_conf}<br>
      <b>Persistent bias:</b> {_fmt(row.get('persistent_station_bias_cent_l'), 1, ' c/L')}<br>
      <b>Persistence:</b> {persistence}<br>
      <b>Updated:</b> {updated}<br>
    </div>
    """


def _legend_html() -> str:
    items = "".join(
        f'<div><span style="display:inline-block;width:14px;height:14px;background:{color};'
        f'border:1px solid #777;margin-right:7px;vertical-align:-2px"></span>{label}</div>'
        for _, _, color, label in _COLOR_BINS
    )
    return f"""
    <div style="position:fixed;bottom:28px;left:28px;z-index:9999;background:white;
                border:1px solid #bbb;border-radius:8px;padding:10px 12px;
                box-shadow:0 1px 6px rgba(0,0,0,.2);font:12px/1.55 Arial,sans-serif">
      <b>Local price excess</b><br>{items}
      <div style="margin-top:5px"><span style="display:inline-block;width:14px;height:14px;
      background:#9e9e9e;border:1px solid #777;margin-right:7px;vertical-align:-2px"></span>
      low confidence / stale / unassessed</div>
      <div style="margin-top:7px;color:#666;max-width:230px">Price anomaly indicator only — not a legal finding.</div>
    </div>
    """


def build_station_map(live_scores: pd.DataFrame, output: str | Path) -> Path:
    """Build an interactive Leaflet/Folium station map for the v0.6.1 dashboard."""
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - clear runtime message
        raise RuntimeError(
            "folium is required for the v0.6.1 station map. Run `python -m pip install -e .`."
        ) from exc

    work = live_scores.copy()
    work["fuel"] = work["fuel"].astype(str).str.upper()
    work["_lat"], work["_lon"] = _coordinates(work)
    work["local_excess_cent_l"] = pd.to_numeric(
        work.get("local_excess_cent_l"), errors="coerce"
    )
    work["price_age_days"] = pd.to_numeric(work.get("price_age_days"), errors="coerce")
    work = work[
        work["_lat"].between(41.0, 52.0)
        & work["_lon"].between(-6.0, 10.0)
    ].copy()

    assessed = work.get("peer_confidence", pd.Series(index=work.index, dtype="object")).isin(
        ["HIGH", "MEDIUM"]
    )
    fresh = work["price_age_days"].le(30) | work["price_age_days"].isna()
    has_excess = work["local_excess_cent_l"].notna()
    work["_robust"] = assessed & fresh & has_excess

    m = folium.Map(
        location=[46.6, 2.2],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    for fuel in ("GAZOLE", "SP95"):
        layer = folium.FeatureGroup(
            name=f"{fuel} — assessed",
            show=(fuel == "GAZOLE"),
        )
        subset = work[(work["fuel"] == fuel) & work["_robust"]]
        for _, row in subset.iterrows():
            color = local_excess_color(row["local_excess_cent_l"])
            folium.CircleMarker(
                location=[float(row["_lat"]), float(row["_lon"])],
                radius=4,
                color="#555555",
                weight=0.45,
                fill=True,
                fill_color=color,
                fill_opacity=0.82,
                tooltip=f"{fuel} · {row.get('ville', '')} · {_fmt(row.get('local_excess_cent_l'), 1, ' c/L')}",
                popup=folium.Popup(_popup(row), max_width=330),
            ).add_to(layer)
        layer.add_to(m)

    uncertain = folium.FeatureGroup(name="Low confidence / stale / unassessed", show=False)
    for _, row in work[~work["_robust"]].iterrows():
        folium.CircleMarker(
            location=[float(row["_lat"]), float(row["_lon"])],
            radius=3,
            color="#777777",
            weight=0.4,
            fill=True,
            fill_color="#9e9e9e",
            fill_opacity=0.35,
            tooltip=f"{row.get('fuel', '')} · {row.get('ville', '')} · low-confidence/unassessed",
            popup=folium.Popup(_popup(row), max_width=330),
        ).add_to(uncertain)
    uncertain.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend_html()))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output
