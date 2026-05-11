"""Mastering others is strength; mastering yourself is true power. — Lao Tzu"""

from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd

from ._containers import DescriptiveResult


class _TableParser(HTMLParser):
    """Parse all <table> elements from HTML into lists of rows."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell = ""
        self._current_table: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self._in_row:
            self._in_cell = True
            self._current_cell = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._in_cell:
            self._current_row.append(self._current_cell.strip())
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._in_row = False
        elif tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell += data


def extract_tables(html: str) -> list[pd.DataFrame]:
    """Parse all ``<table>`` elements from HTML into DataFrames.

    The first row of each table is used as column headers.

    Parameters
    ----------
    html : str

    Returns
    -------
    list[pd.DataFrame]
    """
    parser = _TableParser()
    parser.feed(html)
    frames = []
    for tbl in parser.tables:
        if len(tbl) < 2:
            frames.append(pd.DataFrame(tbl))
        else:
            header = tbl[0]
            data = tbl[1:]
            frames.append(pd.DataFrame(data, columns=header))
    return frames


def extract_tables_result(html: str) -> DescriptiveResult:
    """Wrapper returning a DescriptiveResult."""
    tables = extract_tables(html)
    return DescriptiveResult(
        name="HTML tables",
        value=len(tables),
        extra={"tables": tables},
    )


tblxt = extract_tables_result


def cheatsheet() -> str:
    return "Mastering others is strength; mastering yourself is true power. — Lao Tzu"
