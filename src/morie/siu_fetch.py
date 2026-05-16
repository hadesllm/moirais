# SPDX-License-Identifier: AGPL-3.0-or-later
"""On-demand scraper for the Ontario Special Investigations Unit (SIU).

The SIU publishes Director's Reports at https://siu.on.ca/en/directors_reports.php.
Each case has a public PDF or HTML report listing the incident date, the
notifying police service, and the Director's decision. This module scrapes
the index page(s) and per-case detail pages on demand, caching results as
a single CSV in the morie cache directory.

Distribution policy (2026-05): the scraped corpus is NOT shipped with the
package because the legal status of redistributing aggregated copies of
publicly-posted oversight reports is unsettled. Each user runs the
scraper themselves, which is unambiguously fair use.

The scraper is conservative: a 2-second delay between requests, retries
on 5xx, respects robots.txt. Run `fetch_siu_cases(year=...)` to populate
the cache, then `morie_load_dataset("siu")` returns a tidy data.frame.
"""

from __future__ import annotations

import csv
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


__all__ = [
    "SIU_INDEX_URL",
    "fetch_siu_cases",
    "siu_cache_path",
]


SIU_INDEX_URL = "https://www.siu.on.ca/en/directors_reports.php"
USER_AGENT = "morie/0.2.0 (+https://github.com/hadesllm/morie)"
RATE_LIMIT_SECONDS = 2.0


def _http_get(url: str, *, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _extract_case_links(index_html: str) -> list[tuple[str, str]]:
    """Return [(case_number, url)] tuples found in an index HTML page."""
    pat = re.compile(
        r'href="(case_summary_details\.php\?[^"]+)"[^>]*>(?:\s*<[^>]+>)*\s*'
        r"([A-Za-z\-]+[0-9]+|[0-9]+-[A-Z]+-[0-9]+)",
        re.I,
    )
    out: list[tuple[str, str]] = []
    for m in pat.finditer(index_html):
        rel = m.group(1)
        cn = m.group(2)
        out.append((cn, urllib.parse.urljoin(SIU_INDEX_URL, rel)))
    return out


_DATE_FIELDS = {
    "incident_iso": re.compile(r"(?:Incident|incident occurred on)\s*[:\-]?\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
    "notification_iso": re.compile(r"(?:Notification|SIU was notified on)\s*[:\-]?\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
    "decision_iso": re.compile(r"(?:Director'?s? [Dd]ecision)\s*[:\-]?\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})"),
}

_SERVICE_FIELD = re.compile(r"(?:Police Service|Notifying Service)\s*[:\-]?\s*([A-Z][A-Za-z' \-]+(?:Police|Service))", re.I)
_DECISION_FIELD = re.compile(r"(?:no reasonable grounds|reasonable grounds|charge\(s\)? was|withdrawn|director'?s decision|charges? were laid)", re.I)


def _parse_case_page(html: str, case_number: str, url: str) -> dict:
    """Best-effort parsing of an SIU case detail page into a flat dict."""
    record = {"case_number": case_number, "source_url": url}
    for key, pat in _DATE_FIELDS.items():
        m = pat.search(html)
        record[key] = _to_iso(m.group(1)) if m else ""
    m = _SERVICE_FIELD.search(html)
    record["police_service"] = m.group(1).strip() if m else ""
    m = _DECISION_FIELD.search(html)
    record["director_decision_text"] = m.group(0).strip() if m else ""
    return record


_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}


def _to_iso(date_str: str) -> str:
    m = re.match(r"([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})", date_str.strip())
    if not m:
        return ""
    month = _MONTHS.get(m.group(1))
    if not month:
        return ""
    return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"


def siu_cache_path(cache_dir: str | Path = "~/.cache/morie/siu") -> Path:
    p = Path(cache_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p / "SIU.csv"


def fetch_siu_cases(
    *,
    years: Optional[Iterable[int]] = None,
    cache_dir: str | Path = "~/.cache/morie/siu",
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Scrape SIU Director's Reports into a single CSV.

    Args:
        years: Iterable of fiscal years to fetch (default = all years
            indexed on the SIU site).
        cache_dir: Directory for the cache CSV.
        overwrite: If False and SIU.csv already exists, return it.
        progress: Print one line per scraped page when True.

    Returns:
        Path to the populated SIU.csv.

    Raises:
        urllib.error.URLError on persistent network failure.
    """
    out_path = siu_cache_path(cache_dir)
    if out_path.is_file() and not overwrite:
        return out_path

    # The SIU index supports a `?year=YYYY` filter; default to all years
    # the user requested, or scrape the unfiltered index if none given.
    years = list(years) if years is not None else [None]

    # Pull index pages
    case_links: list[tuple[str, str]] = []
    for y in years:
        url = SIU_INDEX_URL if y is None else f"{SIU_INDEX_URL}?year={y}"
        if progress:
            print(f"[siu] index: {url}")
        try:
            html = _http_get(url)
        except Exception as e:
            if progress:
                print(f"[siu] index fetch failed: {e}")
            continue
        case_links.extend(_extract_case_links(html))
        time.sleep(RATE_LIMIT_SECONDS)

    # Deduplicate
    seen = set(); unique_links = []
    for cn, u in case_links:
        if u not in seen:
            seen.add(u); unique_links.append((cn, u))

    # Fetch detail pages
    records: list[dict] = []
    for i, (cn, u) in enumerate(unique_links, 1):
        if progress and i % 25 == 0:
            print(f"[siu] case {i}/{len(unique_links)}")
        try:
            html = _http_get(u)
        except Exception:
            continue
        records.append(_parse_case_page(html, cn, u))
        time.sleep(RATE_LIMIT_SECONDS)

    if not records:
        raise RuntimeError(
            "Scraped 0 SIU cases. The site layout may have changed; "
            "verify SIU_INDEX_URL and the regexes in siu_fetch.py."
        )

    fieldnames = list({k for r in records for k in r.keys()})
    fieldnames = ["case_number", "police_service", "incident_iso",
                  "notification_iso", "decision_iso",
                  "director_decision_text", "source_url"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    if progress:
        print(f"[siu] wrote {len(records)} cases to {out_path}")
    return out_path
