"""Special Investigations Unit (SIU) director's-report mining.

The Ontario Special Investigations Unit publishes the Director's
final reports for every investigated incident at
https://www.siu.on.ca/en/director_reports.php as PDFs.  Each report
follows a stable structural template (date, location, allegation,
narrative, conclusion) and contains the named subjects involved.

This module provides three layers:

  1. :func:`list_reports`        — fetch the index page and return a
                                    DataFrame of (report_id, url,
                                    incident_date, location, allegation).
  2. :func:`fetch_report_text`   — download a single report PDF and
                                    return the extracted plain text
                                    (via pypdf, already a soft dep).
  3. :func:`extract_report_fields` — apply the SIU-template regex to
                                    a report text and return a dict
                                    of structured fields.

The fields produced by step 3 are the inputs the MRM modules expect
when running SIU-side analyses.

Quick usage
-----------

  >>> from morie.ingest.siu import list_reports, fetch_report_text, extract_report_fields
  >>> idx = list_reports()
  >>> idx.head()
       report_id  incident_date  ... allegation                              url
  ...
  >>> text = fetch_report_text(idx.iloc[0]["url"])
  >>> fields = extract_report_fields(text)
  >>> fields["conclusion"][:120]
  "The Director ... finds that there are no reasonable grounds ..."

CLI
---

::

    morie ingest siu --list                       # index → CSV
    morie ingest siu --report-id 22-OFD-001 \\
                     --out reports/22-OFD-001/    # text + fields

"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd

DEFAULT_USER_AGENT = "morie/0.7.4 (+https://github.com/hadesllm/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0

# SIU site layout (verified 2026-05-13)
SIU_BASE = "https://www.siu.on.ca"
SIU_INDEX = f"{SIU_BASE}/en/directors_reports.php"


class SIUError(RuntimeError):
    """An SIU fetch / parse failed."""


# ----------------------------------------------------------------------
# Step 1 — list the published reports


def list_reports(*, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 user_agent: str = DEFAULT_USER_AGENT) -> pd.DataFrame:
    """Fetch the SIU director's-reports index and parse it.

    **Known limitation (v0.5.0):** the SIU re-launched their website
    in 2025 with a JS-rendered case list — the index page at
    ``/en/directors_reports.php`` is a thin shell, and the actual
    case data is loaded dynamically via JavaScript.  Until we add
    a proper headless-browser fetch (or reverse-engineer the
    AJAX endpoint), this function returns the **legacy** index
    page's PDF anchors only, which means many recent reports
    will be missing.  Use :func:`fetch_report_text` directly with
    a known PDF URL — that path is unaffected by the redesign.

    Returns an empty DataFrame with the canonical schema if no
    PDF anchors are found.

    Returns a DataFrame with columns:
      - ``report_id``       : e.g. "22-OFD-001"
      - ``url``             : direct PDF URL
      - ``incident_date``   : ISO date string when known
      - ``location``        : city / region string
      - ``allegation``      : short allegation label
      - ``release_date``    : date the report was posted (may be NaN)
    """
    import warnings

    headers = {"User-Agent": user_agent}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        r = c.get(SIU_INDEX)
        if r.status_code >= 400:
            raise SIUError(f"index page -> HTTP {r.status_code}")
        html = r.text

    # bs4 is a hard dep
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    rows: list[dict[str, Any]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        # Resolve relative URL
        url = urljoin(SIU_BASE, href)
        # report id is typically the first chunk of the filename
        fname = href.rsplit("/", 1)[-1]
        rid = fname.rsplit(".", 1)[0]
        # Surrounding text in the parent <tr> / <li> often carries
        # incident date + location + allegation; try to scrape it.
        parent = a.find_parent(["tr", "li", "p", "div"])
        ptext = parent.get_text(" ", strip=True) if parent else ""
        rows.append({
            "report_id": rid,
            "url": url,
            "release_date": _scrape_date(ptext),
            "incident_date": _scrape_incident_date(ptext),
            "location": _scrape_location(ptext),
            "allegation": _scrape_allegation(ptext),
        })
    if not rows:
        warnings.warn(
            "siu.list_reports() found zero PDF anchors on the SIU "
            "index page.  The SIU re-launched their site in 2025 with "
            "a JS-rendered case list; the legacy index pattern this "
            "function uses no longer surfaces recent reports.  "
            "Pass a known PDF URL directly to fetch_report_text().",
            UserWarning,
            stacklevel=2,
        )
    # Always return the canonical schema, even if empty
    return pd.DataFrame(
        rows,
        columns=["report_id", "url", "release_date", "incident_date", "location", "allegation"],
    )


# ----------------------------------------------------------------------
# Step 2 — fetch a single report's text


def fetch_report_text(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                      user_agent: str = DEFAULT_USER_AGENT) -> str:
    """Download a single SIU director's-report PDF and extract its text.

    Uses pypdf (a soft dep declared in ``pyproject.toml`` ``test`` extras
    and hard-loaded here).  Returns the concatenated text of every page.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SIUError(
            "siu.fetch_report_text() needs pypdf — install with "
            "`pip install 'morie[test]'` or `pip install pypdf`"
        ) from exc

    headers = {"User-Agent": user_agent}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        r = c.get(url)
        if r.status_code >= 400:
            raise SIUError(f"report fetch -> HTTP {r.status_code}: {url}")
        pdf_bytes = r.content

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — individual page failures shouldn't kill the report
            parts.append("")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Step 3 — pull structured fields from the report text


# SIU director's-report template (stable across 2018-2026)
_SECTION_HEADINGS = {
    "summary": re.compile(r"\b(Summary of the Incident|Summary of Incident|Summary)\s*[:\.\n]", re.IGNORECASE),
    "investigation": re.compile(r"\b(The Investigation|The Investigative Action)\s*[:\.\n]", re.IGNORECASE),
    "narrative": re.compile(r"\b(Narrative of Events|Narrative)\s*[:\.\n]", re.IGNORECASE),
    "evidence": re.compile(r"\b(Evidence)\s*[:\.\n]", re.IGNORECASE),
    "law": re.compile(r"\b(Relevant Legislation|Applicable Law)\s*[:\.\n]", re.IGNORECASE),
    "analysis": re.compile(r"\b(Analysis|Analysis and Director's Decision)\s*[:\.\n]", re.IGNORECASE),
    "conclusion": re.compile(r"\b(Decision|Conclusion|Director's Decision)\s*[:\.\n]", re.IGNORECASE),
}

_REPORT_ID = re.compile(r"\b(\d{2}-[A-Z]{3,4}-\d{3,4})\b")
_INCIDENT_DATE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}",
    re.IGNORECASE,
)


def extract_report_fields(text: str) -> dict[str, Any]:
    """Apply the SIU-template regex to a report and return structured fields.

    Returned dict (every key may be missing or None on parse failure):
      - ``report_id``        : 22-OFD-001 style
      - ``incident_date``    : first detected long-form date
      - ``sections``         : dict mapping section name -> raw text slice
      - ``conclusion``       : the Director's-decision section, isolated
      - ``raw_length``       : len(text), for sanity checking
    """
    out: dict[str, Any] = {
        "report_id": None,
        "incident_date": None,
        "sections": {},
        "conclusion": None,
        "raw_length": len(text),
    }

    m = _REPORT_ID.search(text)
    if m:
        out["report_id"] = m.group(1)

    m = _INCIDENT_DATE.search(text)
    if m:
        out["incident_date"] = m.group(0)

    # Section slicing — find each heading, take text up to the next heading
    boundaries: list[tuple[str, int]] = []
    for name, pat in _SECTION_HEADINGS.items():
        m = pat.search(text)
        if m:
            boundaries.append((name, m.end()))
    boundaries.sort(key=lambda x: x[1])
    for i, (name, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
        # back off to the start of the next section's heading
        if i + 1 < len(boundaries):
            next_pat = _SECTION_HEADINGS[boundaries[i + 1][0]]
            m = next_pat.search(text, pos=start)
            if m:
                end = m.start()
        out["sections"][name] = text[start:end].strip()
    if "conclusion" in out["sections"]:
        out["conclusion"] = out["sections"]["conclusion"]

    return out


# ----------------------------------------------------------------------
# Internal scrapers for the index-page row text


_DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})|(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b)", re.IGNORECASE)


def _scrape_date(text: str) -> str | None:
    m = _DATE_PAT.search(text)
    if not m:
        return None
    return next((g for g in m.groups() if g), None)


def _scrape_incident_date(text: str) -> str | None:
    # SIU index sometimes shows "Incident: <date>" — grab a second date if present
    matches = _DATE_PAT.findall(text)
    if len(matches) >= 2:
        # second match is usually incident date
        m = matches[1]
        return next((g for g in m if g), None)
    return _scrape_date(text)


def _scrape_location(text: str) -> str | None:
    # Heuristic: capitalised city words near "in <City>"
    m = re.search(r"\bin\s+([A-Z][a-zA-Z' \-]+(?:,\s*Ontario)?)", text)
    return m.group(1).strip() if m else None


def _scrape_allegation(text: str) -> str | None:
    # Heuristic: SIU uses short allegation labels — Custody Death, Firearm Discharge, etc.
    m = re.search(
        r"\b(Custody Death|Custody Injury|Firearm Discharge|Sexual Assault Allegation|"
        r"Vehicle Pursuit|Vehicle Injury|Vehicle Death|Other)\b",
        text,
        re.IGNORECASE,
    )
    return m.group(1).title() if m else None


# ----------------------------------------------------------------------
# CLI handler — wired into morie/runner.py


def cli(args: list[str]) -> int:
    """Handle ``morie ingest siu ...``.  Returns exit code."""
    import argparse
    import json
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="morie ingest siu",
                                description="Pull SIU director's-report index or a single report.")
    p.add_argument("--list", action="store_true", help="Fetch the index page and emit CSV to stdout")
    p.add_argument("--report-id", help="Report id (e.g. 22-OFD-001); requires --out")
    p.add_argument("--url", help="Direct PDF URL of a single report")
    p.add_argument("--out", type=Path, help="Output directory for the fetched report's text + fields")
    ns = p.parse_args(args)

    if ns.list:
        df = list_reports()
        sys.stdout.write(df.to_csv(index=False))
        return 0

    url = ns.url
    if url is None and ns.report_id:
        # Resolve from index
        df = list_reports()
        match = df[df["report_id"] == ns.report_id]
        if match.empty:
            sys.stderr.write(f"report id {ns.report_id!r} not found in current index\n")
            return 3
        url = match.iloc[0]["url"]
    if url is None:
        p.error("provide --list, --report-id, or --url")
        return 2

    if not ns.out:
        p.error("--out <dir> is required when fetching a report")
        return 2
    ns.out.mkdir(parents=True, exist_ok=True)

    text = fetch_report_text(url)
    fields = extract_report_fields(text)
    (ns.out / "report.txt").write_text(text)
    (ns.out / "fields.json").write_text(json.dumps(fields, indent=2))
    sys.stderr.write(f"wrote {ns.out / 'report.txt'} ({len(text):,} chars) + fields.json\n")
    return 0
