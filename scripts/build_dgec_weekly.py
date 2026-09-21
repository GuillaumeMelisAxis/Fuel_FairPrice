from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import pandas as pd
import pdfplumber
import requests


BASE_URL = (
    "https://www.ecologie.gouv.fr/sites/default/files/documents/"
    "NPG-{date}.pdf"
)


def _fr_number(value: str | None) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("\xa0", " ").replace(" ", "").replace(",", ".")

    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    return float(text)


def _find_row_numbers(page, label: str) -> list[float]:
    """
    Find a table row whose text contains `label`, returning numeric cells that
    follow the label cell. Works with the stable tabular layout used by NPG.
    """
    target = label.casefold()

    for table in page.extract_tables():
        if not table:
            continue

        for row in table:
            if not row:
                continue

            cells = ["" if c is None else str(c).strip() for c in row]

            label_index = None
            for i, cell in enumerate(cells):
                if target in cell.casefold():
                    label_index = i
                    break

            if label_index is None:
                continue

            numbers = []
            for cell in cells[label_index + 1 :]:
                number = _fr_number(cell)
                if number is not None:
                    numbers.append(number)

            if numbers:
                return numbers

    return []


def parse_npg_pdf(content: bytes, report_date: pd.Timestamp) -> list[dict]:
    """
    Extract the CURRENT weekly observation from one NPG bulletin.

    Stable layout checked against NPG-2026.09.11:
      page index 2: refined product quotations in USD/t
      page index 4: national weekly HTT prices in c€/L
    """
    with pdfplumber.open(BytesIO(content)) as pdf:
        if len(pdf.pages) < 5:
            raise ValueError("Unexpected NPG document: fewer than 5 pages")

        quote_page = pdf.pages[2]
        htt_page = pdf.pages[4]

        euro_nums = _find_row_numbers(quote_page, "Eurosuper")
        diesel_nums = _find_row_numbers(quote_page, "Gazole")

        sp95_htt_nums = _find_row_numbers(htt_page, "sp95-e5")
        diesel_htt_nums = _find_row_numbers(htt_page, "gazole")

        if not euro_nums or not diesel_nums:
            raise ValueError("Could not parse refined quotations")

        if not sp95_htt_nums or not diesel_htt_nums:
            raise ValueError("Could not parse HTT prices")

        return [
            {
                "date": report_date.normalize(),
                "fuel": "SP95",
                "quote_usd_t": euro_nums[0],
                "htt_cents_l": sp95_htt_nums[0],
            },
            {
                "date": report_date.normalize(),
                "fuel": "GAZOLE",
                "quote_usd_t": diesel_nums[0],
                "htt_cents_l": diesel_htt_nums[0],
            },
        ]


def collect_weekly_history(
    start: str,
    end: str,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """
    Try every Friday in [start, end]. Missing bulletins are skipped.
    """
    dates = pd.date_range(start, end, freq="W-FRI")
    rows = []

    session = requests.Session()

    for date in dates:
        url = BASE_URL.format(date=date.strftime("%Y.%m.%d"))

        response = session.get(url, timeout=timeout)

        if response.status_code != 200:
            print(f"SKIP {date.date()} -> HTTP {response.status_code}")
            continue

        try:
            parsed = parse_npg_pdf(response.content, date)
        except Exception as exc:
            print(f"PARSE ERROR {date.date()} -> {exc}")
            continue

        rows.extend(parsed)
        print(f"OK {date.date()}")

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return (
        out.drop_duplicates(["date", "fuel"])
        .sort_values(["date", "fuel"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    output = ROOT / "config" / "dgec_weekly_market.csv"

    df = collect_weekly_history(
        start="2025-01-03",
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved {len(df)} rows to {output}")
    print(df.tail(10).to_string(index=False))
