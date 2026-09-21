from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from pypdf import PdfReader

NPG_URL_TEMPLATE = (
    "https://www.ecologie.gouv.fr/sites/default/files/documents/"
    "NPG-{issue_date}.pdf"
)

FRENCH_MONTHS = {
    "janv": 1,
    "jan": 1,
    "févr": 2,
    "fevr": 2,
    "fév": 2,
    "fev": 2,
    "mars": 3,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "août": 8,
    "aout": 8,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "déc": 12,
    "dec": 12,
}


@dataclass(frozen=True)
class NpgObservation:
    date: pd.Timestamp
    fuel: str
    france_htt_eur_l: float
    refined_quote_eur_l: float
    issue_date: pd.Timestamp
    source_url: str

    @property
    def observed_distribution_margin_eur_l(self) -> float:
        return self.france_htt_eur_l - self.refined_quote_eur_l


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _parse_month_token(token: str, issue_date: date) -> pd.Timestamp:
    token = token.lower().replace(".", "").strip()
    m = re.match(r"([a-zàâäéèêëîïôöùûüç]+)[-\s]?(\d{2}|\d{4})", token)
    if not m:
        raise ValueError(f"Cannot parse French month token: {token!r}")
    month_name, year_token = m.groups()
    month_key = next((k for k in FRENCH_MONTHS if month_name.startswith(k)), None)
    if month_key is None:
        raise ValueError(f"Unknown French month in token: {token!r}")
    year = int(year_token)
    if year < 100:
        year += 2000
    return pd.Timestamp(year=year, month=FRENCH_MONTHS[month_key], day=1)


def _extract_section(text: str, start_markers: tuple[str, ...], end_markers: tuple[str, ...]) -> str:
    lower = text.lower()
    starts = [lower.find(m.lower()) for m in start_markers]
    starts = [p for p in starts if p >= 0]
    if not starts:
        return ""
    start = min(starts)
    ends = [lower.find(m.lower(), start + 1) for m in end_markers]
    ends = [p for p in ends if p >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end]


def _parse_pair(section: str, fuel: str, issue_date: date, source_url: str) -> list[NpgObservation]:
    section = _normalise_text(section)
    # The DGEC table contains: month1 month2, then France values and Cotations values.
    month_tokens = re.findall(
        r"(?:janv|févr|fevr|mars|avr|mai|juin|juil|août|aout|sept|oct|nov|déc|dec)\.?(?:-|\s)?\d{2,4}",
        section,
        flags=re.IGNORECASE,
    )
    # Deduplicate while preserving order.
    month_tokens = list(dict.fromkeys(month_tokens))
    if len(month_tokens) < 2:
        return []

    number = r"(\d{1,3}[,.]\d{1,3})"
    france_match = re.search(rf"(?:France|ance)\s*\|?\s*{number}\s*\|?\s*{number}", section, re.IGNORECASE)
    quote_match = re.search(rf"(?:Cotations|Cotation|tations)\s*\|?\s*{number}\s*\|?\s*{number}", section, re.IGNORECASE)
    if not france_match or not quote_match:
        return []

    dates = [_parse_month_token(t, issue_date) for t in month_tokens[:2]]
    france = [float(v.replace(",", ".")) / 100.0 for v in france_match.groups()]
    quote = [float(v.replace(",", ".")) / 100.0 for v in quote_match.groups()]

    return [
        NpgObservation(
            date=d,
            fuel=fuel,
            france_htt_eur_l=h,
            refined_quote_eur_l=q,
            issue_date=pd.Timestamp(issue_date),
            source_url=source_url,
        )
        for d, h, q in zip(dates, france, quote)
    ]


def parse_npg_text(text: str, issue_date: date, source_url: str = "") -> pd.DataFrame:
    """Parse current/previous monthly SP95-E5 and Gazole tables from an NPG PDF text."""
    sp95 = _extract_section(
        text,
        ("Supercarburant sans plomb (sp95-e5)", "sp95-e5"),
        ("Gazole en c€/l", "Gazole"),
    )
    diesel = _extract_section(
        text,
        ("Gazole en c€/l", "Gazole"),
        ("Fioul domestique",),
    )
    observations = _parse_pair(sp95, "SP95", issue_date, source_url)
    observations += _parse_pair(diesel, "GAZOLE", issue_date, source_url)

    rows = [
        {
            "date": o.date,
            "fuel": o.fuel,
            "france_htt_eur_l": o.france_htt_eur_l,
            "refined_quote_eur_l": o.refined_quote_eur_l,
            "observed_distribution_margin_eur_l": o.observed_distribution_margin_eur_l,
            "issue_date": o.issue_date,
            "source_url": o.source_url,
        }
        for o in observations
    ]
    return pd.DataFrame(rows)


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_npg(issue_date: date, timeout: int = 30, cache_dir: str | Path | None = None) -> pd.DataFrame:
    issue_str = issue_date.strftime("%Y.%m.%d")
    url = NPG_URL_TEMPLATE.format(issue_date=issue_str)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"NPG-{issue_str}.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 404:
            return pd.DataFrame()
        response.raise_for_status()
        content = response.content
        if cache_path is not None:
            cache_path.write_bytes(content)

    text = extract_pdf_text(content)
    return parse_npg_text(text, issue_date=issue_date, source_url=url)


def fridays_in_month(year: int, month: int) -> list[date]:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, 1)
    out: list[date] = []
    while d.day <= last_day:
        if d.weekday() == 4:
            out.append(d)
        d += timedelta(days=1)
    return out


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthBegin(1)).normalize()


def collect_finalized_months(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timeout: int = 30,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Collect finalized monthly quotations efficiently from DGEC NPG bulletins.

    For target month M, the collector tries Friday issues in M+1.  The previous-
    month column in those bulletins is treated as the finalized observation.  This
    avoids using an incomplete within-month average and requires roughly one PDF
    per month rather than every weekly bulletin.
    """
    start_ts = pd.Timestamp(start).to_period("M").to_timestamp()
    end_ts = pd.Timestamp(end).to_period("M").to_timestamp()
    target_months = pd.date_range(start_ts, end_ts, freq="MS")
    collected: list[pd.DataFrame] = []

    for target in target_months:
        publication_month = next_month(target)
        found: pd.DataFrame | None = None
        for issue in fridays_in_month(publication_month.year, publication_month.month):
            try:
                df = fetch_npg(issue, timeout=timeout, cache_dir=cache_dir)
            except (requests.RequestException, ValueError):
                continue
            if df.empty:
                continue
            candidate = df[df["date"] == target].copy()
            if len(candidate) >= 2:
                candidate["source_status"] = "finalized_previous_month"
                found = candidate
                break
        if found is not None:
            collected.append(found)

    if not collected:
        return pd.DataFrame(
            columns=[
                "date", "fuel", "france_htt_eur_l", "refined_quote_eur_l",
                "observed_distribution_margin_eur_l", "issue_date", "source_url",
                "source_status",
            ]
        )

    out = pd.concat(collected, ignore_index=True)
    return out.sort_values(["date", "fuel"]).reset_index(drop=True)
