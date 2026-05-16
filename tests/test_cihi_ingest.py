# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.ingest.cihi -- the CIHI indicator-table ingest.

The data-sheet picker is tested with synthetic workbooks; no live
CIHI download is performed here.
"""
import io

import pandas as pd

from morie.ingest.cihi import _pick_data_sheet


def _workbook(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return pd.ExcelFile(buf)


def test_pick_data_sheet_skips_small_notes_sheet():
    notes = pd.DataFrame({"Notes": ["intro", "method", "caveats"]})
    data = pd.DataFrame({c: range(50) for c in "wxyz"})
    xl = _workbook({"Notes": notes, "Data": data})
    assert _pick_data_sheet(xl).shape == (50, 4)


def test_pick_data_sheet_single_sheet():
    only = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    xl = _workbook({"Sheet1": only})
    assert _pick_data_sheet(xl).shape == (2, 2)


def test_cihi_entries_registered():
    from morie.data import DATASET_CATALOG
    for key in ("cihi820a", "cihi820b", "cihi849", "cihi885a", "cihi885b"):
        entry = DATASET_CATALOG[key]
        assert entry["fetcher"] == "morie.ingest.cihi:fetch_cihi_xlsx"
        assert entry["fetcher_args"]["url"].endswith(".xlsx")
