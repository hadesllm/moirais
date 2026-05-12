"""Field-level normalisation helpers for SIU rows.

Each helper returns the (iso_or_canonical_value, raw) pair so callers can
keep the verbatim string and a normalised one side-by-side.
"""

from __future__ import annotations

import re
from datetime import datetime, date
from typing import Optional


# Case Number canonical pattern: YY-XXX-NNN  (e.g. 17-PVI-371, 21-TCI-045)
# 2-digit year, 2-4 letter SIU region/team code, 2-4 digit sequence.
CASE_NUMBER_RE = re.compile(r"\b(\d{2})-([A-Z]{2,4})-(\d{2,4})\b")

# Tolerant prefix variants found in the wild on siu.on.ca:
#   "Case Number: 17-PVI-371"
#   "Case # 17-PVI-371"
#   "Case#17-PVI-371"
#   "SIU Case Number 17-PVI-371"
CASE_NUMBER_LABEL_RE = re.compile(
    r"(?:SIU\s+)?Case\s*(?:Number|#)?\s*[:#]?\s*(\d{2}-[A-Z]{2,4}-\d{2,4})",
    re.IGNORECASE,
)


def find_case_number(text: str) -> Optional[str]:
    """Extract canonical Case Number from arbitrary page text.

    Tries the labeled form first ("Case Number: …") then the bare regex
    match. Returns None if no plausible case number is found.
    """
    m = CASE_NUMBER_LABEL_RE.search(text)
    if m:
        return m.group(1).upper()
    m = CASE_NUMBER_RE.search(text)
    if m:
        yy, code, seq = m.group(1), m.group(2), m.group(3)
        return f"{yy}-{code.upper()}-{seq}"
    return None


# Date parsing -- multiple formats seen on siu.on.ca:
#   "January 5, 2017"   "1 January 2017"   "2017-01-05"   "01/05/2017"
DATE_FORMATS = [
    "%B %d, %Y",
    "%d %B, %Y",     # "6 December, 2018" -- SIU news-release stamp
    "%d %B %Y",
    "%B %d %Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%d %b, %Y",
]


def parse_date(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Parse a date string. Returns (iso_string, raw_string).

    Both elements are None if `text` is falsy or unparseable. The raw
    element is *always* preserved exactly as input (after surrounding
    whitespace strip) so audit consumers can replay parsing decisions.
    """
    if not text:
        return None, None
    raw = text.strip()
    if not raw:
        return None, None
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(raw, fmt).date()
            return d.isoformat(), raw
        except ValueError:
            continue
    # Last-ditch ISO regex
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", raw)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.isoformat(), raw
        except ValueError:
            pass
    return None, raw  # raw preserved, iso unknown


def parse_drid_from_url(url: str) -> Optional[int]:
    m = re.search(r"\bdrid=(\d+)\b", url)
    return int(m.group(1)) if m else None


def parse_nrid_from_url(url: str) -> Optional[int]:
    m = re.search(r"\bnrid=(\d+)\b", url)
    return int(m.group(1)) if m else None


def normalise_sex(text: Optional[str]) -> Optional[str]:
    """Map free-form sex/gender text to {male,female,nonbinary,unknown}."""
    if not text:
        return None
    t = text.strip().lower()
    if t in ("m", "male", "man"):
        return "male"
    if t in ("f", "female", "woman"):
        return "female"
    if any(k in t for k in ("non-binary", "nonbinary", "non binary", "nb")):
        return "nonbinary"
    if any(k in t for k in ("unknown", "n/a", "not specified", "not stated", "n.s.")):
        return "unknown"
    return t  # preserve unmapped value rather than dropping it


def normalise_yes_no(text: Optional[str]) -> Optional[bool]:
    """Map yes/no/true/false/charged/no charges -> bool."""
    if not text:
        return None
    t = text.strip().lower()
    if t in ("yes", "y", "true", "t", "1", "charged"):
        return True
    if t in ("no", "n", "false", "f", "0", "no charges", "not charged"):
        return False
    return None
