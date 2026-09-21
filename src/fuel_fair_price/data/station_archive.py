from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests

ANNUAL_URL = "https://donnees.roulez-eco.fr/opendata/annee/{year}"
CURRENT_YEAR_URL = "https://donnees.roulez-eco.fr/opendata/annee"
FUEL_NAMES = {"SP95": "SP95", "Gazole": "GAZOLE", "GAZOLE": "GAZOLE"}


def annual_url(year: int, current_year: int | None = None) -> str:
    current_year = current_year or pd.Timestamp.today().year
    return CURRENT_YEAR_URL if year == current_year else ANNUAL_URL.format(year=year)


def download_annual_zip(year: int, destination: str | Path, timeout: int = 120) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(annual_url(year), timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _normalise_price(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        value = float(str(raw).replace(",", "."))
    except ValueError:
        return None
    # Older XML vintages used thousandths of EUR; modern feeds use EUR directly.
    if value > 20:
        value /= 1000.0
    if not (0.2 <= value <= 5.0):
        return None
    return value


def _mainland_ex_corsica(cp: str) -> bool:
    digits = "".join(ch for ch in str(cp) if ch.isdigit())
    if len(digits) < 2:
        return False
    dep = digits[:2]
    return dep not in {"20", "97", "98"}


def parse_annual_xml(xml_path: str | Path) -> pd.DataFrame:
    """Parse fuel-price change events from an official annual XML file."""
    records: list[dict] = []
    context = ET.iterparse(str(xml_path), events=("end",))
    for _event, elem in context:
        if elem.tag.lower().endswith("pdv"):
            station_id = elem.attrib.get("id")
            cp = elem.attrib.get("cp", "")
            if not _mainland_ex_corsica(cp):
                elem.clear()
                continue
            pop = elem.attrib.get("pop")
            lat = elem.attrib.get("latitude")
            lon = elem.attrib.get("longitude")
            for price in elem.findall(".//prix"):
                raw_name = price.attrib.get("nom") or price.attrib.get("fuel")
                fuel = FUEL_NAMES.get(str(raw_name))
                if fuel not in {"SP95", "GAZOLE"}:
                    continue
                value = _normalise_price(price.attrib.get("valeur"))
                updated = pd.to_datetime(price.attrib.get("maj"), errors="coerce")
                if value is None or pd.isna(updated):
                    continue
                records.append(
                    {
                        "station_id": station_id,
                        "cp": str(cp),
                        "adresse": elem.findtext("adresse"),
                        "ville": elem.findtext("ville"),
                        "road_type": {"A": "AUTOROUTE", "R": "ROUTE"}.get(pop, "UNKNOWN"),
                        "latitude": float(lat) / 100000.0 if lat else np.nan,
                        "longitude": float(lon) / 100000.0 if lon else np.nan,
                        "fuel": fuel,
                        "updated_at": updated,
                        "price_eur_l": value,
                    }
                )
            elem.clear()
    return pd.DataFrame.from_records(records)


def parse_annual_zip(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise ValueError(f"No XML file found in {zip_path}")
        xml_name = max(xml_names, key=lambda n: zf.getinfo(n).file_size)
        zf.extract(xml_name, path=tmp)
        return parse_annual_xml(Path(tmp) / xml_name)


def month_end_station_snapshot(events: pd.DataFrame, year: int) -> pd.DataFrame:
    """Reconstruct month-end station prices from price-change events.

    Pass events from year-1 and year when available so January can carry forward
    a station's last price from the previous December.
    """
    if events.empty:
        return pd.DataFrame()
    ev = events.copy().sort_values(["station_id", "fuel", "updated_at"])
    end_of_year = pd.Timestamp(f"{year}-12-31")
    max_event = pd.Timestamp(ev["updated_at"].max()).normalize()
    last_complete_month = (max_event.to_period("M") - 1).to_timestamp("M") if max_event.year == year else end_of_year
    cutoff = min(end_of_year, last_complete_month)
    month_ends = pd.date_range(f"{year}-01-31", cutoff, freq="ME")
    outputs: list[pd.DataFrame] = []

    for month_end in month_ends:
        eligible = ev[ev["updated_at"] <= month_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
        if eligible.empty:
            continue
        snapshot = eligible.groupby(["station_id", "fuel"], as_index=False).tail(1).copy()
        snapshot["date"] = month_end.to_period("M").to_timestamp()
        outputs.append(snapshot)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def monthly_station_benchmark(snapshot: pd.DataFrame) -> pd.DataFrame:
    """National monthly median and cross-sectional dispersion of station prices."""
    if snapshot.empty:
        return pd.DataFrame()
    return (
        snapshot.groupby(["date", "fuel"], as_index=False)
        .agg(
            station_median_eur_l=("price_eur_l", "median"),
            station_mean_eur_l=("price_eur_l", "mean"),
            station_p10_eur_l=("price_eur_l", lambda s: s.quantile(0.10)),
            station_p90_eur_l=("price_eur_l", lambda s: s.quantile(0.90)),
            station_count=("station_id", "nunique"),
        )
    )
