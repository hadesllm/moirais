"""SIU Ontario report HTML parser. 'Our greatest glory is not in never falling, but in rising every time we fall. — Confucius'

from __future__ import annotations

import re
from html.parser import HTMLParser

from ._containers import DescriptiveResult


class _SIUParser(HTMLParser):
    """Extract structured fields from SIU report HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, str] = {}
        self._current_tag = ""
        self._current_class = ""
        self._capture = False
        self._buffer = ""
        self._all_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._current_tag = tag
        attr_dict = dict(attrs)
        self._current_class = attr_dict.get("class", "")

    def handle_data(self, data: str) -> None:
        self._all_text.append(data.strip())

    def get_all_text(self) -> str:
        return " ".join(t for t in self._all_text if t)


def siu_scrape_report(html: str) -> DescriptiveResult:
    """Parse an SIU Ontario report page to extract key fields.

    Extracts case_number, date, subject_officer, allegations, and findings
    by searching the text for known patterns.

    Parameters
    ----------
    html : str
        Raw HTML of an SIU report page.

    Returns
    -------
    DescriptiveResult
        ``value`` is a dict with extracted fields.
    """
    parser = _SIUParser()
    parser.feed(html)
    text = parser.get_all_text()

    case_match = re.search(r"(?:Case|File)\s*(?:No\.?|Number|#)\s*[:.]?\s*([\w\-/]+)", text, re.IGNORECASE)
    date_match = re.search(r"(?:Date|Dated)\s*[:.]?\s*(\w+\s+\d{1,2},?\s+\d{4})", text, re.IGNORECASE)
    officer_match = re.search(r"(?:Subject\s+Officer|Officer)\s*[:.]?\s*([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", text)
    allegation_match = re.search(r"(?:Allegation|Charge)s?\s*[:.]?\s*(.+?)(?:\.|$)", text, re.IGNORECASE)
    finding_match = re.search(r"(?:Finding|Conclusion|Decision)s?\s*[:.]?\s*(.+?)(?:\.|$)", text, re.IGNORECASE)

    result = {
        "case_number": case_match.group(1).strip() if case_match else None,
        "date": date_match.group(1).strip() if date_match else None,
        "subject_officer": officer_match.group(1).strip() if officer_match else None,
        "allegations": allegation_match.group(1).strip() if allegation_match else None,
        "findings": finding_match.group(1).strip() if finding_match else None,
    }
    return DescriptiveResult(name="SIU report", value=result)


siubc = siu_scrape_report


def cheatsheet() -> str:
    return "siu_scrape_report({}) -> SIU Ontario report HTML parser. 'In my experience there is n"
