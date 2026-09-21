from __future__ import annotations

from io import BytesIO
from urllib.parse import urljoin
import re

import pandas as pd
import requests


EU_BULLETIN_PAGE = (
    "https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en"
)


def _discover_history_url(timeout: float = 30.0) -> str:
    """
    Discover the current EU Commission 'Price developments 2005 onwards'
    workbook instead of hard-coding a document UUID.
    """
    response = requests.get(EU_BULLETIN_PAGE, timeout=timeout)
    response.raise_for_status()

    html = response.text

    # The current workbook filename contains
    # Weekly_Oil_Bulletin_Prices_History...xlsx
    matches = re.findall(
        r'href=["\']([^"\']*Weekly[^"\']*Oil[^"\']*Bulletin[^"\']*'
        r'Prices[^"\']*History[^"\']*\.xlsx[^"\']*)["\']',
        html,
        flags=re.IGNORECASE,
    )

    if not matches:
        # More tolerant fallback: any .xlsx link containing History.
        matches = re.findall(
            r'href=["\']([^"\']*History[^"\']*\.xlsx[^"\']*)["\']',
            html,
            flags=re.IGNORECASE,
        )

    if not matches:
        raise RuntimeError(
            "Could not discover the EU Weekly Oil Bulletin history workbook."
        )

    return urljoin(EU_BULLETIN_PAGE, matches[0].replace("&amp;", "&"))


def download_history_workbook(timeout: float = 60.0) -> bytes:
    url = _discover_history_url(timeout=timeout)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    if len(response.content) < 100_000:
        raise RuntimeError(
            f"EU workbook download looks too small ({len(response.content)} bytes)."
        )

    return response.content


def _clean_name(value) -> str:
    value = "" if pd.isna(value) else str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _parse_mixed_date(series: pd.Series) -> pd.Series:
    # Try pandas' general parser first.
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)

    # Excel serial fallback.
    numeric = pd.to_numeric(series, errors="coerce")
    mask = parsed.isna() & numeric.notna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            numeric.loc[mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return parsed


def parse_price_sheet(
    workbook: bytes,
    sheet_name: str = "Prices wo taxes",
) -> dict[str, pd.DataFrame]:
    """
    Parse the EU workbook country blocks.

    Workbook layout:
      - first column = date
      - first row contains repeated 'CTR' columns
      - each CTR starts one country block
      - first 3 rows are headers/metadata
    """
    raw = pd.read_excel(
        BytesIO(workbook),
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
    )

    header = raw.iloc[0].astype(str).tolist()
    header[0] = "date"

    ctr_cols = [
        i for i, value in enumerate(header)
        if str(value).strip() == "CTR"
    ]

    if not ctr_cols:
        raise RuntimeError(
            f"No CTR blocks found in EU sheet {sheet_name!r}."
        )

    end_cols = ctr_cols[1:] + [raw.shape[1]]
    result: dict[str, pd.DataFrame] = {}

    for start, end in zip(ctr_cols, end_cols):
        # Include global date column plus this country block.
        tmp = pd.concat(
            [
                raw.iloc[:, [0]],
                raw.iloc[:, start:end],
            ],
            axis=1,
        ).copy()

        block_headers = ["date", "ctr"]
        block_headers.extend(
            _clean_name(x)
            for x in header[start + 1:end]
        )

        # Ensure unique names.
        used = {}
        unique_headers = []
        for name in block_headers:
            base = name or "unnamed"
            count = used.get(base, 0)
            unique_headers.append(
                base if count == 0 else f"{base}_{count}"
            )
            used[base] = count + 1

        tmp.columns = unique_headers

        # Skip workbook's three header rows.
        tmp = tmp.iloc[3:].copy()

        country_values = (
            tmp["ctr"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        if country_values.empty:
            continue

        country = country_values.iloc[0].rstrip("_")
        tmp = tmp.drop(columns=["ctr"])

        tmp["date"] = _parse_mixed_date(tmp["date"])

        for col in tmp.columns:
            if col != "date":
                tmp[col] = pd.to_numeric(
                    tmp[col],
                    errors="coerce",
                )

        tmp = (
            tmp.dropna(subset=["date"])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )

        tmp = tmp.dropna(axis=1, how="all")
        result[country] = tmp

    return result


def _find_product_column(columns, *, gasoline: bool) -> str:
    cols = [str(c) for c in columns if c != "date"]

    if gasoline:
        priorities = (
            ("euro", "95"),
            ("gasoline", "95"),
            ("petrol", "95"),
            ("eurosuper",),
        )
    else:
        priorities = (
            ("gas", "oil", "auto"),
            ("diesel",),
            ("automotive",),
            ("gasoil", "auto"),
        )

    for terms in priorities:
        for col in cols:
            low = col.lower()
            if all(term in low for term in terms):
                return col

    raise RuntimeError(
        "Could not identify product column. "
        f"Available columns: {cols}"
    )


def load_france_weekly_htt(
    *,
    start: str | None = "2020-01-01",
) -> pd.DataFrame:
    """
    Return France weekly prices without taxes (HTT) for:
      - SP95
      - GAZOLE

    EU workbook units are EUR / 1000 litres.
    """
    workbook = download_history_workbook()
    countries = parse_price_sheet(
        workbook,
        sheet_name="Prices wo taxes",
    )

    # Workbook uses country codes; France is normally FR.
    france = countries.get("FR")

    if france is None:
        candidates = [
            k for k in countries
            if str(k).upper().startswith("FR")
        ]
        if not candidates:
            raise RuntimeError(
                f"France block not found. Country codes: {sorted(countries)[:40]}"
            )
        france = countries[candidates[0]]

    sp95_col = _find_product_column(
        france.columns,
        gasoline=True,
    )
    diesel_col = _find_product_column(
        france.columns,
        gasoline=False,
    )

    rows = []

    for fuel, col in (
        ("SP95", sp95_col),
        ("GAZOLE", diesel_col),
    ):
        tmp = france[["date", col]].copy()
        tmp = tmp.rename(columns={col: "htt_eur_1000l"})
        tmp["fuel"] = fuel

        # Source unit: EUR / 1000 L.
        tmp["htt_eur_l"] = (
            tmp["htt_eur_1000l"] / 1000.0
        )

        rows.append(
            tmp[["date", "fuel", "htt_eur_l"]]
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["htt_eur_l"])

    if start is not None:
        out = out[
            out["date"] >= pd.Timestamp(start)
        ]

    return (
        out.sort_values(["fuel", "date"])
        .reset_index(drop=True)
    )
