"""Parse a single SIU director's-report HTML page into a structured row.

This module is **pure** -- no network. Hand it a raw HTML string and it
returns a dict matching SIU_COLUMNS. All scraping, retry, and caching
lives in `_scraper.py`.

The actual SIU page markup is subject to change; this parser hardens
extraction by:
  - looking for several label variants per field
  - falling back to regex on stripped-text when DOM structure shifts
  - preserving the verbatim `narrative_full` regardless of parse success
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup

from ._schema import BLANK_ROW
from ._normalize import (
    find_case_number,
    parse_date,
    parse_drid_from_url,
    parse_nrid_from_url,
    normalise_sex,
    normalise_yes_no,
)

PARSER_VERSION = "0.1.0"


def parse_html(html: str, *, drid: Optional[int] = None,
               source_url: Optional[str] = None) -> dict:
    """Turn a director's-report HTML page into a SIU.csv row dict.

    :param html: raw response body.
    :param drid: optional integer drid from the request URL -- useful when
                 the page itself doesn't echo it.
    :param source_url: optional canonical URL -- used to derive drid/nrid
                       and recorded as `source_url_report`.
    :return: dict with every SIU_COLUMNS key (Nones for unfound fields).
    """
    soup = BeautifulSoup(html, "lxml")
    full_text = _stripped_text(soup)
    text = _trim_to_body(full_text)
    # Use full_text for case-number search (page header may be in chrome)
    # but body-only for everything else.

    row = dict(BLANK_ROW)
    row["parser_version"] = PARSER_VERSION
    row["source_url_report"] = source_url

    # ── identifiers ────────────────────────────────────────────────
    row["case_number"] = find_case_number(full_text)

    # ── language detection (en/fr) ─────────────────────────────────
    # The SIU site publishes some reports French-only on the /en/ URL.
    # Phase 2b finding (2026-05-06): drid=4000 (Case 23-OCD-100) is
    # French-only despite the /en/ path. We mark the row's language
    # and skip English-only structured extraction for French pages.
    lang = _detect_language(text)
    row["_language"] = lang
    if lang == "fr":
        # Skip the rest of the structured English extraction.
        # narrative_full + supplemental + case_number are already populated.
        row["narrative_full"] = _extract_narrative_full(soup, text)
        row["narrative_summary"] = _extract_summary(text)
        row["supplemental_materials"] = _extract_outbound_links(soup, source_url)
        # nrid lookup still works (URL-based, language-agnostic)
        nrid_link = _find_news_release_link(soup, source_url)
        if nrid_link:
            row["nrid"] = parse_nrid_from_url(nrid_link)
            row["source_url_news"] = nrid_link
        if drid is None and source_url:
            drid = parse_drid_from_url(source_url)
        row["drid"] = drid
        return row

    if drid is None and source_url:
        drid = parse_drid_from_url(source_url)
    row["drid"] = drid

    # nrid: look for any link to news_template.php?nrid=… on the page
    nrid_link = _find_news_release_link(soup, source_url)
    if nrid_link:
        row["nrid"] = parse_nrid_from_url(nrid_link)
        row["source_url_news"] = nrid_link

    # ── narrative (always preserved regardless of structured-parse) ─
    row["narrative_full"] = _extract_narrative_full(soup, text)
    row["narrative_summary"] = _extract_summary(text)

    # ── police service (prose-detected; full names + abbreviations).
    # Vocabulary scan only -- DO NOT fall through to `_label_value`
    # because the SIU page boilerplate contains the substring "police
    # services across Ontario" which the label regex matches as
    # garbage like "s across Ontario.".
    row["police_service"] = _detect_police_service(text)

    # ── The Team subsection: officer / investigator counts ─────────
    team_section = _section_text(text, "The Team", end_markers=("Complainant", "Civilian Witnesses"))
    row["number_of_officers_involved"] = _detect_officers_involved(text, team_section)

    # The Team also has the SIU investigator counts (kept as
    # supplemental -- "siu_investigators" gets a list-as-string).
    row["siu_investigators"] = _team_field(team_section, "SIU Investigators")
    row["siu_forensics_investigators"] = _team_field(team_section, "SIU Forensic Investigators")

    # ── witnesses + officer counts (parsed from sub-sections) ──────
    cw_section = _section_text(text, "Civilian Witnesses",
                                end_markers=("Witness Officers", "Subject Officers", "Incident Narrative"))
    row["number_of_civilian_witnesses"] = _count_lines_starting_with(cw_section, "CW")

    # Modern (post-SIU Act 2019) format calls these "Witness Officials"
    # and "Subject Official" (singular). Try both in turn.
    wo_section = (
        _section_text(text, "Witness Officials",
                      end_markers=("Evidence", "Subject Official",
                                   "Subject Officers", "Incident Narrative"))
        or _section_text(text, "Witness Officers",
                         end_markers=("Subject Officers", "Subject Official",
                                      "Incident Narrative"))
    )
    row["number_of_witness_officials"] = _detect_witness_officer_count(wo_section)

    so_section = (
        _section_text(text, "Subject Officials",
                      end_markers=("Witness Officials", "Witness Officers",
                                   "Incident Narrative", "Evidence",
                                   "Relevant Legislation"))
        or _section_text(text, "Subject Official",
                         end_markers=("Witness Officials", "Witness Officers",
                                      "Incident Narrative", "Evidence",
                                      "Relevant Legislation"))
        or _section_text(text, "Subject Officers",
                         end_markers=("Incident Narrative", "Evidence",
                                      "Relevant Legislation"))
    )
    row["number_of_subject_officials"] = (
        _count_lines_starting_with(so_section, "The SO")
        or _count_lines_starting_with(so_section, "SO")
    )
    if so_section and any(kw in so_section.lower() for kw in
                          ("interviewed", "declined", "notes received")):
        row["subject_official_interviewed_or_notes"] = "Yes"

    # ── reason / location (prose-mined) ────────────────────────────
    row["location_of_call"] = (
        _label_value(text, "Location")
        or _detect_location_from_intro(text)
    )
    row["reason_for_interaction"] = _label_value(text, "Reason for Interaction")

    # ── date_of_incident ───────────────────────────────────────────
    raw_inc = (
        _label_value(text, "Date of Incident")
        or _label_value(text, "Incident Date")
        or _detect_incident_date_from_narrative(text)
    )
    iso_inc, raw_inc_kept = parse_date(raw_inc)
    row["date_of_incident_iso"], row["date_of_incident_raw"] = iso_inc, raw_inc_kept

    # ── date_siu_notified (the "Notification of the SIU" subsection) ─
    raw_not = (
        _label_value(text, "Date SIU was Notified")
        or _label_value(text, "SIU Notified")
        or _detect_siu_notified(text)
    )
    iso_not, raw_not_kept = parse_date(raw_not)
    row["date_siu_notified_iso"], row["date_siu_notified_raw"] = iso_not, raw_not_kept

    # ── notifying party (prose: "the X notified the SIU") ─────────
    row["notifying_party"] = _detect_notifying_party(text)

    # ── director's decision date (signature block "Date: <date>") ──
    raw_dec = (
        _label_value(text, "Date of SIU Director's Decision")
        or _label_value(text, "Decision Date")
        or _detect_decision_date(text)
    )
    iso_dec, raw_dec_kept = parse_date(raw_dec)
    row["date_of_director_decision_iso"], row["date_of_director_decision_raw"] = iso_dec, raw_dec_kept

    # ── injuries, sex, age ────────────────────────────────────────
    row["injuries_sustained"] = (
        _label_value(text, "Injuries Sustained")
        or _detect_injuries_sustained(text)
    )
    row["specific_injuries"] = _detect_specific_injuries(text)

    sex, age = _detect_age_sex(text)
    row["sex_gender_affected"] = normalise_sex(sex) if sex else \
        normalise_sex(_label_value(text, "Sex") or _label_value(text, "Gender"))
    row["age_affected"] = age or _label_value(text, "Age")

    # ── relevant legislation (parse the named "Section X, ACT" hits) ─
    row["relevant_legislation"] = _detect_legislation(text)

    # ── decision verdict ──────────────────────────────────────────
    row["charges_recommended"] = normalise_yes_no(
        _label_value(text, "Charges Recommended") or _label_value(text, "Charges Laid")
    )
    if row["charges_recommended"] is None:
        row["charges_recommended"] = _detect_charges_from_decision(text)
    row["directors_decision_reasonable"] = _detect_reasonable_grounds(text)

    # ── mental-health / race signal ────────────────────────────────
    row["mental_health_or_race_indications"] = _scan_mental_health_race(row["narrative_full"] or "")

    # ── supplemental links ────────────────────────────────────────
    row["supplemental_materials"] = _extract_outbound_links(soup, source_url)

    return row


def parse_news_html(html: str, *, nrid: Optional[int] = None,
                    source_url: Optional[str] = None) -> dict:
    """Parse a SIU news-release page into a partial row dict.

    News-release pages live at `news_template.php?nrid=<N>` and have a
    different layout than the director's reports -- single headline,
    short summary paragraph, signed-by-Director line. We extract:
      - news_release_title
      - news_release_date_iso / _raw       (the "Mississauga, ON (Date)" line)
      - news_release_summary               (1-2 paragraphs)
      - case_number                         (cross-validation with drid scrape)
      - nrid                                (URL locator)
      - directors_name                      (e.g. "Tony Loparco")

    The caller (typically `scrape_drid` after finding the nrid link)
    merges these into the main row dict by case_number match.
    """
    soup = BeautifulSoup(html, "lxml")
    full_text = _stripped_text(soup)

    out = {
        "nrid": nrid,
        "source_url_news": source_url,
        "news_release_title": None,
        "news_release_date_iso": None,
        "news_release_date_raw": None,
        "news_release_summary": None,
        "case_number": find_case_number(full_text),
        "directors_name": None,
    }

    # ── Find the headline. The body always has "News Release\n<title>\n"
    # -- find the SECOND occurrence (first is the page chrome menu).
    title = _detect_news_release_title(full_text)
    out["news_release_title"] = title

    # ── Release date: "<City>, ON (DD Month, YYYY) ---"
    iso, raw = _detect_news_release_date(full_text)
    out["news_release_date_iso"] = iso
    out["news_release_date_raw"] = raw

    # ── Summary paragraph (the "On <date>, …" sentence right after the date stamp)
    out["news_release_summary"] = _detect_news_release_summary(full_text)

    # ── Director's name -- "The Director of the Special Investigations Unit, Tony Loparco,"
    out["directors_name"] = _detect_directors_name(full_text)

    return out


# ── extraction primitives ─────────────────────────────────────────────────

def _stripped_text(soup: BeautifulSoup) -> str:
    """Get the page's visible text with normalised whitespace."""
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    txt = soup.get_text("\n", strip=True)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt


# Section names that appear in the "Contents:" TOC at the top of every
# SIU page. We skip past them by anchoring on the second occurrence of
# "Mandate of the SIU" (which is always the body section header).
def _trim_to_body(text: str) -> str:
    """Slice past the TOC into the actual report body.

    Strategy: find the 2nd occurrence of "Mandate of the SIU" (1st is
    in Contents: TOC). If only one exists, use it. Anything before the
    body is navigation chrome / TOC / boilerplate.
    """
    matches = list(re.finditer(r"(?:^|\n)\s*Mandate of the SIU\s*\n", text, re.M))
    if len(matches) >= 2:
        return text[matches[1].start():]
    if len(matches) == 1:
        return text[matches[0].start():]
    # No "Mandate of the SIU" -- fall back to "The Investigation"
    matches = list(re.finditer(r"(?:^|\n)\s*The Investigation\s*\n", text, re.M))
    if len(matches) >= 2:
        return text[matches[1].start():]
    return text


def _label_value(text: str, label: str) -> Optional[str]:
    """Look for `<label>:?\\s*<value>` in the stripped text and return the value.

    Tolerant of label variants -- the caller passes the canonical phrase;
    we accept it case-insensitively with optional trailing colon and
    whitespace flexibility.
    """
    pat = re.compile(
        rf"{re.escape(label)}\s*[:\-]?\s*(.{{1,200}}?)(?=\n|$)",
        re.IGNORECASE,
    )
    m = pat.search(text)
    if not m:
        return None
    val = m.group(1).strip().rstrip(",;:")
    return val or None


def _label_int(text: str, label: str) -> Optional[int]:
    raw = _label_value(text, label)
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    return int(m.group(0)) if m else None


def _find_news_release_link(soup: BeautifulSoup,
                            source_url: Optional[str]) -> Optional[str]:
    """Return absolute URL of any `news_template.php?nrid=…` link on the page."""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "news_template.php" in href and "nrid=" in href:
            if source_url and not href.startswith("http"):
                href = urljoin(source_url, href)
            return href
    return None


_NARRATIVE_HEADERS = (
    "The Investigation", "Investigation", "Director's Decision",
    "The Director's Decision", "Brief Summary", "Narrative",
)
_NARRATIVE_START_RE = re.compile(
    r"(?:Mandate engaged|The Investigation|Notification of the SIU)",
    re.IGNORECASE,
)
_NARRATIVE_END_RE = re.compile(
    r"(?:Endnotes|News Releases for this Case|Note:\s*\n*The signed English|"
    r"^\s*THE UNIT\s*$|^\s*PROGRAMS AND SERVICES\s*$)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_narrative_full(soup: BeautifulSoup, text: str) -> str:
    """Return the body of the report -- Mandate engaged -> Endnotes.

    The SIU page is wrapped in a navigation chrome that pollutes both
    `<main>` and a flat `<p>` join with the site menu (which contains
    "First Nations, Inuit and Métis Liaison Program" and other phrases
    that would falsely trigger the MH/race scanner). We slice the body
    explicitly to start at "Mandate engaged" / "The Investigation" and
    end at "Endnotes" / "News Releases for this Case".
    """
    # First try DOM containers (real production HTML may have one).
    for selector in ("div.report-body", "div.article-body",
                     "article", "div#main-content"):
        node = soup.select_one(selector)
        if node:
            return node.get_text("\n", strip=True)
    # Fall back to text-slice approach
    start_m = _NARRATIVE_START_RE.search(text)
    end_m = _NARRATIVE_END_RE.search(text)
    start = start_m.start() if start_m else 0
    end = end_m.start() if end_m else len(text)
    if end > start:
        return text[start:end]
    # Last-ditch -- flatten <p> tags
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join(p for p in paras if p)


def _extract_summary(text: str) -> Optional[str]:
    """Pull the first meaningful paragraph as a summary. Limited to 1500 chars."""
    paras = [p for p in text.split("\n\n") if len(p) > 80]
    if not paras:
        return None
    return paras[0][:1500]


_MH_RACE_KEYWORDS = (
    # Mental health
    "mental health", "mental illness", "psychiatric", "schizophren",
    "bipolar", "depression", "psychosis", "psychotic", "suicidal",
    "self-harm", "delusion", "crisis intervention", "EDP",
    "emotionally disturbed", "MCIT",
    # Race
    "Black", "African", "Indigenous", "First Nations", "Métis", "Metis",
    "Inuit", "racialised", "racialized", "racial", "anti-Black",
    "anti-black", "ethnic", "South Asian", "racism",
)


def _scan_mental_health_race(narrative: str) -> str:
    """Return a `;`-joined list of matched MH/race keywords (or empty)."""
    if not narrative:
        return ""
    found: list[str] = []
    low = narrative.lower()
    for kw in _MH_RACE_KEYWORDS:
        if kw.lower() in low and kw not in found:
            found.append(kw)
    return "; ".join(found)


def _extract_outbound_links(soup: BeautifulSoup,
                            source_url: Optional[str]) -> str:
    """Joined string of absolute URLs to non-siu.on.ca pages on the report."""
    base_host = urlparse(source_url).netloc if source_url else "siu.on.ca"
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        if source_url and not href.startswith("http"):
            href = urljoin(source_url, href)
        host = urlparse(href).netloc
        if host and host != base_host and "siu.on.ca" not in host:
            if href not in out:
                out.append(href)
    return "; ".join(out)


# ── Prose-mining helpers ───────────────────────────────────────────────────
# The real SIU report HTML is mostly narrative paragraphs with section
# headers. These helpers find structured information embedded in prose.

# Police service vocabulary -- every singleton in SIU1a's Q3 column.
# Compiled 2026-05-06 from data/datasets/vsr/SIU1a.xlsx (90 distinct
# values, 56 unique singletons). Multi-service comma-joined entries are
# handled at match-time, not in this list.
_POLICE_SERVICES = [
    "Toronto Police Service",
    "RCMP",
    "Royal Canadian Mounted Police",
    "Ontario Provincial Police",
    "Peel Regional Police",
    "Vancouver Police Department",
    "Niagara Regional Police Service",
    "Hamilton Police Service",
    "Ottawa Police Service",
    "York Regional Police",
    "London Police Service",
    "Durham Regional Police Service",
    "Windsor Police Service",
    "Waterloo Regional Police Service",
    "Halton Regional Police Service",
    "Brantford Police Service",
    "Barrie Police Service",
    "Greater Sudbury Police Service",
    "Victoria/Esquimalt Police Department",
    "Kingston Police Service",
    "Guelph Police Service",
    "Peterborough Police Service",
    "Thunder Bay Police Service",
    "Chatham-Kent Police Service",
    "North Bay Police Service",
    "Sarnia Police Service",
    "Sault Ste. Marie Police Service",
    "Belleville Police Service",
    "Timmins Police Service",
    "Abbotsford Police Department",
    "Cornwall Community Police Service",
    "Cornwall Police Service",      # actual page form (SIU1a sometimes entered "Community" version)
    "St. Thomas Police Service",
    "Hanover Police Service",
    "Stratford Police Service",
    "Kawartha Lakes Police Service",
    "Saanich Police Department",
    "South Simcoe Police Service",
    "Owen Sound Police Service",
    "New Westminster Police Service",
    "Cobourg Police Service",
    "Niagara Parks Commission",
    "Dryden Police Service",
    "Brockville Police Service",
    "Delta Police Department",
    "West Vancouver Police Department",
    "Port Moody Police Department",
    "Gananoque Police Service",
    "Smith Falls Police Service",
    "Saugeen Shores Police Service",
    "Nelson Police Department",
    "Surrey Police Department",
    "Woodstock Police Service",
    "Port Hope Police Service",
    "Stl'atl'imx Tribal Police",
    "West Grey Police Service",
    "Anishinabek Police Service",
    "Six Nations Police",
    "Akwesasne Mohawk Police Service",
    "Nishnawbe Aski Police Service",
]

# Abbreviations seen in modern (post-2019) reports. Standalone-word match.
_POLICE_ABBR = {
    "OPP": "Ontario Provincial Police",
    "TPS": "Toronto Police Service",
    "RCMP": "RCMP",
    "PRP": "Peel Regional Police",
    "VPD": "Vancouver Police Department",
    "NRPS": "Niagara Regional Police Service",
    "HPS": "Hamilton Police Service",
    "YRP": "York Regional Police",
    "LPS": "London Police Service",
    "DRPS": "Durham Regional Police Service",
    "WPS": "Windsor Police Service",
    "WRPS": "Waterloo Regional Police Service",
    "HRPS": "Halton Regional Police Service",
    "BPS": "Brantford Police Service",
    "GSPS": "Greater Sudbury Police Service",
    "TBPS": "Thunder Bay Police Service",
    "SSMPS": "Sault Ste. Marie Police Service",
    "GPS": "Guelph Police Service",
    "CKPS": "Chatham-Kent Police Service",
    "NBPS": "North Bay Police Service",
}


# Modern SIU report boilerplate -- the jurisdictional sidebar/footer
# mentions Niagara Parks Commission ~4× per page in EVERY case (because
# NPC is one of the entities under SIU's jurisdiction). Pre-strip these
# boilerplate clauses before vocabulary scan.
_SIU_NPC_BOILERPLATE_RE = re.compile(
    r"(?:special\s+constables?\s+of\s+the\s+|"
    r"a\s+special\s+constable\s+of\s+the\s+)"
    r"Niagara\s+Parks\s+Commission"
    r"(?:\s+(?:and|or)\s+(?:a\s+)?peace\s+officers?\s+(?:with|under)\s+"
    r"(?:the\s+)?Legisla[a-z]*[^.]{0,120})?",
    re.IGNORECASE,
)


def _detect_police_service(text: str) -> Optional[str]:
    """Find the canonical police-service name mentioned most in `text`.

    Counts occurrences of each vocabulary entry and returns the
    most-frequent. This handles multi-service cases where a primary
    and a witness service both appear -- SIU1a captures the primary
    (the most-mentioned one).

    Falls back to abbreviation match if no full-name hits.

    Boilerplate-aware: strips known jurisdictional boilerplate
    occurrences so the SIU's "About" footer doesn't false-tag every
    modern report as a Niagara Parks Commission case.
    """
    # Strip NPC boilerplate before vocabulary scan
    cleaned = _SIU_NPC_BOILERPLATE_RE.sub("", text)
    counts: dict[str, int] = {}
    for name in _POLICE_SERVICES:
        c = cleaned.count(name)
        if c > 0:
            counts[name] = c

    if counts:
        # If RCMP and "Royal Canadian Mounted Police" both hit, prefer RCMP
        # (canonical SIU1a label) by collapsing.
        if "Royal Canadian Mounted Police" in counts and "RCMP" in counts:
            counts.pop("Royal Canadian Mounted Police")
        elif "Royal Canadian Mounted Police" in counts and "RCMP" not in counts:
            counts["RCMP"] = counts.pop("Royal Canadian Mounted Police")
        # Highest count wins; ties broken by longer name (more specific)
        return max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

    # No full-name hit -- try abbreviations as standalone tokens
    abbr_counts: dict[str, int] = {}
    for abbr, full in _POLICE_ABBR.items():
        c = len(re.findall(rf"\b{re.escape(abbr)}\b", text))
        if c > 0:
            abbr_counts[full] = abbr_counts.get(full, 0) + c
    if abbr_counts:
        return max(abbr_counts.items(), key=lambda kv: kv[1])[0]
    return None


def _section_text(text: str, header: str,
                  end_markers: tuple[str, ...] = ()) -> str:
    """Return the text starting at `header` up to the first end_marker.

    Used to slice out subsections like "The Team", "Subject Officers",
    "Civilian Witnesses". Header match is case-sensitive (SIU pages are
    consistent) but tolerant of leading/trailing whitespace.
    """
    pat = re.compile(rf"(?:^|\n)\s*{re.escape(header)}\s*\n", re.M)
    m = pat.search(text)
    if not m:
        return ""
    start = m.end()
    # Find earliest end_marker after `start`
    end = len(text)
    for em in end_markers:
        em_pat = re.compile(rf"(?:^|\n)\s*{re.escape(em)}\s*\n", re.M)
        em_m = em_pat.search(text, start)
        if em_m and em_m.start() < end:
            end = em_m.start()
    return text[start:end]


def _team_field(team_section: str, label: str) -> Optional[str]:
    """Pull a field like `Number of SIU Investigators assigned: 2`."""
    if not team_section:
        return None
    pat = re.compile(rf"Number of {re.escape(label)} assigned\s*:\s*(\d+)", re.IGNORECASE)
    m = pat.search(team_section)
    return m.group(1) if m else None


def _detect_officers_involved(text: str, team_section: str) -> Optional[int]:
    """Detect officers involved in the incident (NOT SIU investigators).

    Order of attempts:
      1. Explicit `Number of Officers: N` label (legacy/test format)
      2. Numbered `SO #N` mentions in Subject Officers section (real)
      3. Bare "the SO" mention -> assume 1 officer (real, single-SO case)
    """
    # 1) explicit-label form (synthetic / structured pages)
    by_label = _label_int(text, "Number of Officers")
    if by_label is not None:
        return by_label

    # 2) numbered SOs in Subject Officers section. Note: real SIU HTML
    # breaks `SO\n#1` across lines, so flatten whitespace first.
    so_section = _section_text(text, "Subject Officers",
                                end_markers=("Incident Narrative", "Evidence"))
    haystack = so_section or text
    flat = re.sub(r"\s+", " ", haystack)
    so_nums = set()
    for m in re.finditer(r"\bSO\s*#\s*(\d+)\b", flat):
        so_nums.add(int(m.group(1)))
    if so_nums:
        return max(so_nums)

    # 3) bare "the SO" -> 1
    if re.search(r"\bthe\s+SO\b", flat, re.IGNORECASE):
        return 1
    return None


def _count_lines_starting_with(section: str, prefix: str) -> Optional[int]:
    """Count distinct numbered tags like CW #1, CW #2, …

    The real SIU HTML breaks `CW #1` across two lines (`CW` then `#1`),
    so we collapse the section to a single line first then count via
    the numbered pattern. Falls back to counting bare prefix-lines when
    no numbers are found (e.g. a single "The SO" without numbering).
    """
    if not section:
        return None
    flat = re.sub(r"\s+", " ", section)
    nums = set()
    for m in re.finditer(rf"\b{re.escape(prefix)}\s*#?\s*(\d+)\b", flat):
        nums.add(int(m.group(1)))
    if nums:
        return max(nums)
    # Fall back to "The SO" / "the SO" without a number -> 1 if present
    if re.search(rf"\b{re.escape(prefix.strip())}\b", flat):
        return 1
    return None


def _detect_witness_officer_count(wo_section: str) -> Optional[int]:
    """Witness Officers -- usually 'There were no police officers witness…'
    or a list of WO #1, WO #2.

    Real SIU HTML breaks `WO\\n#1` across lines; flatten before regex.
    """
    if not wo_section:
        return None
    flat = re.sub(r"\s+", " ", wo_section)
    if re.search(r"no police officers? witness", flat, re.IGNORECASE):
        return 0
    wo_nums = set()
    for m in re.finditer(r"\bWO\s*#\s*(\d+)\b", flat):
        wo_nums.add(int(m.group(1)))
    return max(wo_nums) if wo_nums else None


def _detect_location_from_intro(text: str) -> Optional[str]:
    """Find "in the [Township|City|Town] of <Place>" pattern early in the
    Investigation section."""
    inv = _section_text(text, "The Investigation", end_markers=("The Team", "Incident Narrative"))
    haystack = inv or text
    m = re.search(r"in the (Township|City|Town|Municipality|Region) of ([A-Z][A-Za-z\s\-]+?)(?:[\.,]|\s+(?:on|at|when))",
                  haystack)
    if m:
        return f"{m.group(1)} of {m.group(2).strip()}"
    return None


def _detect_incident_date_from_narrative(text: str) -> Optional[str]:
    """Find the date of incident inside "Incident Narrative" or "The
    Investigation" sections.

    Skip dates that appear in the same sentence as
    "notified", "contacted", "Notification of the SIU" -- those are
    notification dates, not incident dates. Take the next available
    "On <Month> <Day>, <Year>" instead.
    """
    SKIP_NEAR = ("notified", "contacted the siu", "Notification of the SIU")

    for sec_name in ("Incident Narrative", "The Investigation"):
        sec = _section_text(text, sec_name,
                            end_markers=("Nature of Injuries", "Evidence",
                                         "The Team",  # post-2019 separator
                                         "Analysis and Director",
                                         "Relevant Legislation"))
        if not sec:
            continue
        # Walk every "On <Date>" hit; pick first NOT near a notification verb
        for m in re.finditer(r"\b[Oo]n\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", sec):
            window = sec[max(0, m.start()-50):m.end()+80].lower()
            if any(skip in window for skip in SKIP_NEAR):
                continue
            return m.group(1).replace(",", "")
    return None


def _detect_siu_notified(text: str) -> Optional[str]:
    """Date SIU was notified.

    Two phrasings seen in the wild:
      pre-2019 format: "the OPP notified the SIU"
      post-2019 format: "NRPS contacted the SIU"
    """
    inv = _section_text(text, "The Investigation",
                        end_markers=("The Team", "Incident Narrative"))
    haystack = inv or text
    # Form A: "<Date>, ... notified|contacted the SIU" -- date may be
    # preceded by "On" (capitalized) or "on" (lowercase, often after
    # "at <time> on <weekday>, <date>"); accept either.
    m = re.search(
        r"\b[Oo]n\s+(?:[A-Z][a-z]+,?\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})"
        r"[^\n]{0,200}?(?:notified|contacted)\s+the\s+SIU",
        haystack)
    if m:
        return m.group(1).replace(",", "")
    # Form B: "...notified|contacted the SIU on <Date>"
    m = re.search(
        r"(?:notified|contacted)\s+the\s+SIU[^\n]{0,200}?[Oo]n\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        haystack)
    if m:
        return m.group(1).replace(",", "")
    # Form C: any "On <Date>" inside the "Notification of the SIU"
    # subsection (the modern format opens with this)
    notif = _section_text(text, "Notification of the SIU",
                          end_markers=("The Team", "Incident Narrative",
                                       "Evidence"))
    if notif:
        m = re.search(r"On\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", notif)
        if m:
            return m.group(1).replace(",", "")
    return None


def _detect_notifying_party(text: str) -> Optional[str]:
    """Detect who notified the SIU. Same vocabulary as police service."""
    inv = _section_text(text, "The Investigation",
                        end_markers=("The Team", "Incident Narrative"))
    haystack = inv or text[:5000]
    m = re.search(r"the\s+([A-Z][A-Za-z\s]+?)\s+(?:\([A-Z]+\)\s+)?notified the SIU", haystack)
    if m:
        return m.group(1).strip()
    return _detect_police_service(haystack)


def _detect_decision_date(text: str) -> Optional[str]:
    """Director's decision date -- signature block usually has 'Date: <date>'
    immediately above the director's name."""
    # "Date: October 18, 2018" near "Director" mention.
    m = re.search(r"Date:\s*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", text)
    if m:
        return m.group(1).replace(",", "")
    # ISO date variant
    m = re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    return None


def _detect_age_sex(text: str) -> tuple[Optional[str], Optional[str]]:
    """Find "<age>-year-old <woman|man|female|male|girl|boy>" patterns,
    typically in the "Mandate engaged" or "Complainant" sections."""
    m = re.search(r"\b(\d{1,3})[\s\-]year[\s\-]old\s+(woman|man|female|male|girl|boy|person|individual|youth|child|adult)\b",
                  text, re.IGNORECASE)
    if not m:
        return None, None
    age = m.group(1)
    sex = m.group(2).lower()
    return sex, age


def _detect_injuries_sustained(text: str) -> Optional[str]:
    """Pull the most-specific injury phrase from "Nature of Injuries /
    Treatment" subsection."""
    sec = _section_text(text, "Nature of Injuries / Treatment",
                        end_markers=("Evidence", "Relevant Legislation",
                                      "Analysis and Director"))
    if not sec:
        sec = _section_text(text, "Nature of Injuries",
                             end_markers=("Evidence", "Relevant Legislation"))
    if sec:
        # First non-empty paragraph
        for ln in sec.splitlines():
            s = ln.strip()
            if len(s) > 30:
                return s[:300]
    return None


def _detect_specific_injuries(text: str) -> Optional[str]:
    """Look for fracture/laceration/etc. specifics in narrative."""
    m = re.search(r"((?:fractured?|broken|lacerat\w+|gunshot|stab\w+|burns?)[^\n]{1,200}?(?:rib|leg|arm|skull|wrist|ankle|jaw|nose|tooth|finger|spine|vertebra)[^\n]{0,80})",
                  text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _detect_legislation(text: str) -> Optional[str]:
    """Collect all 'Section N, <Act> – <description>' headings from the
    Relevant Legislation section."""
    sec = _section_text(text, "Relevant Legislation",
                        end_markers=("Analysis and Director", "News Releases"))
    if not sec:
        return None
    sections = []
    for m in re.finditer(r"Section\s+\d+(?:\([^)]+\))?,?\s+([A-Z][^\n,–-]+?)(?:\s*[–-]|$)",
                         sec):
        act = m.group(1).strip().rstrip(",")
        if act not in sections:
            sections.append(act)
    return "; ".join(sections) if sections else None


def _detect_charges_from_decision(text: str) -> Optional[bool]:
    """Read the Analysis and Director's Decision section for the verdict."""
    sec = _section_text(text, "Analysis and Director's Decision",
                        end_markers=("Endnotes", "News Release", "Note:"))
    if not sec:
        return None
    low = sec.lower()
    if any(p in low for p in (
        "no charges", "shall issue", "none shall issue",
        "no basis for charges", "no reasonable grounds",
        "do not lay", "decline to lay", "lack the necessary grounds",
    )):
        return False
    if any(p in low for p in (
        "charged with", "have caused .* to be charged",
        "criminal charges have been laid", "charges have been laid",
    )):
        return True
    return None


# French markers -- strong signals that a page is French-only.
_FR_MARKERS = (
    "L'enquête",
    "L’enquête",
    "Exercice du mandat",
    "Éléments de preuve",
    "Dispositions législatives pertinentes",
    "Témoins civils",
    "Agents impliqués",
    "Mandat de l'UES",
    "Mandat de l’UES",
)
# English markers -- strong signals that a page is English.
_EN_MARKERS = (
    "The Investigation",
    "Notification of the SIU",
    "Mandate engaged",
    "Civilian Witnesses",
    "Witness Officers",
    "Subject Officers",
    "Analysis and Director's Decision",
    "Analysis and Director’s Decision",
)


def _detect_language(text: str) -> str:
    """Return 'en', 'fr', or 'unknown' based on which marker set wins."""
    en_hits = sum(1 for m in _EN_MARKERS if m in text)
    fr_hits = sum(1 for m in _FR_MARKERS if m in text)
    if en_hits >= 2 and en_hits > fr_hits:
        return "en"
    if fr_hits >= 2 and fr_hits > en_hits:
        return "fr"
    return "unknown"


def _detect_news_release_title(full_text: str) -> Optional[str]:
    """Find the headline of a news release.

    The page has chrome menu with "News Release" appearing many times;
    the actual headline appears AFTER the second-occurring "News Release"
    label and BEFORE "Case Number:".
    """
    # Find "News Release" markers and the line after each
    matches = list(re.finditer(r"\n\s*News Release\s*\n", full_text))
    if not matches:
        return None
    # Try each "News Release" anchor -- pick the one followed by a non-empty
    # line that doesn't start with another nav label.
    for m in matches:
        nl = full_text.find("\n", m.end())
        if nl == -1:
            continue
        candidate = full_text[m.end():nl].strip()
        if not candidate:
            continue
        # Reject obvious nav labels
        if candidate.lower() in ("media centre", "news releases", "case numbers"):
            continue
        if len(candidate) < 10 or len(candidate) > 200:
            continue
        return candidate
    return None


_RELEASE_DATE_RE = re.compile(
    r"([A-Za-z\.]+(?:[ \-][A-Za-z\.]+)?,\s*ON)\s*\(\s*(\d{1,2}\s+[A-Z][a-z]+,?\s+\d{4})\s*\)",
    re.IGNORECASE,
)


def _detect_news_release_date(full_text: str) -> tuple[Optional[str], Optional[str]]:
    m = _RELEASE_DATE_RE.search(full_text)
    if not m:
        return None, None
    raw = m.group(2).strip()
    iso, _ = parse_date(raw)
    return iso, raw


def _detect_news_release_summary(full_text: str) -> Optional[str]:
    """Pull the paragraph immediately after the city/date stamp."""
    m = _RELEASE_DATE_RE.search(full_text)
    if not m:
        return None
    after = full_text[m.end():]
    # Strip the trailing "---" if present
    after = re.sub(r"^\s*-+\s*", "", after).lstrip()
    # Take up to the next double-newline or 1500 chars
    chunks = re.split(r"\n\s*\n", after, maxsplit=1)
    if not chunks:
        return None
    return chunks[0].strip()[:1500] or None


_DIRECTOR_NAME_RE = re.compile(
    r"Director of the Special Investigations Unit,\s*([A-Z][A-Za-z'\-]+\s+[A-Z][A-Za-z'\-]+)",
)


def _detect_directors_name(full_text: str) -> Optional[str]:
    m = _DIRECTOR_NAME_RE.search(full_text)
    if m:
        return m.group(1).strip()
    # Director sign-off line: "<Name>\nDirector\nSpecial Investigations Unit"
    m = re.search(r"\n\s*([A-Z][A-Za-z'\-]+\s+[A-Z][A-Za-z'\-]+)\s*\n\s*Director\s*\n",
                   full_text)
    if m:
        return m.group(1).strip()
    return None


def _detect_reasonable_grounds(text: str) -> Optional[str]:
    """Pull the conclusion sentence from the Director's Decision."""
    sec = _section_text(text, "Analysis and Director's Decision",
                        end_markers=("Endnotes", "News Release", "Note:"))
    if not sec:
        return None
    # First sentence containing "reasonable grounds"
    m = re.search(r"([^.\n]*reasonable grounds[^.\n]*\.)", sec, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:400]
    return None
