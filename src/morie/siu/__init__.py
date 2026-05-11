"""morie.siu — automated mining of Ontario Special Investigations Unit
director's reports.

Reads from siu.on.ca (publicly published reports + paired news releases),
parses each report HTML into a structured row, writes a unified SIU.csv
keyed on `case_number` (the SIU's canonical case ID, format YY-XXX-NNN).

URL invariant (see SIU_PLAN_20260506.md):
    drid (URL locator)        ≠ nrid (news-release locator)
    case_number               = canonical join key

Public API:
    scrape_drid(drid, *, client=None, cache=True) -> dict
        Fetch + parse a single director's report by drid.
    scrape_range(drid_min, drid_max, **kw) -> Iterator[dict]
        Polite range scrape with concurrency limits.
    parse_html(html, *, drid=None, source_url=None) -> dict
        Pure parser — no network — turns one HTML page into a row.
    write_csv(rows, path) -> None
    write_jsonl(rows, path) -> None         # for narrative_full bodies
    SIU_COLUMNS                              # 45-col canonical schema
"""

from ._schema import SIU_COLUMNS, BLANK_ROW
from ._scraper import scrape_drid, scrape_range
from ._parser import parse_html, parse_news_html
from ._writer import write_csv, write_jsonl
from . import analyze

__all__ = [
    "SIU_COLUMNS",
    "BLANK_ROW",
    "scrape_drid",
    "scrape_range",
    "parse_html",
    "parse_news_html",
    "write_csv",
    "write_jsonl",
    "analyze",
]
