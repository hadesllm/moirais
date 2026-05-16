# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canadian Institute for Health Information (CIHI) ingest.

CIHI publishes indicator data-table workbooks as ``.xlsx`` files at
stable URLs under ``cihi.ca/sites/default/files/document/...`` (the
"Accessible version" / "data file" link on each indicator page).

:func:`fetch_cihi_xlsx` downloads one and returns a DataFrame.  It is
wired into :data:`morie.data.DATASET_CATALOG` through the ``fetcher``
mechanism, so ``morie.load_dataset()`` can fetch a CIHI indicator
table on demand.
"""
from __future__ import annotations

import io
from typing import Any

__all__ = ["fetch_cihi_xlsx"]


def _pick_data_sheet(xl, **read_excel_kwargs):
    """Return the largest sheet of a workbook as a DataFrame.

    CIHI workbooks lead with a small "Notes"/"Introduction" sheet; the
    data lives on a later, much larger sheet.  Picking the sheet with
    the most cells skips the notes page without hard-coding names.
    """
    best_name, best_df, best_cells = xl.sheet_names[0], None, -1
    for name in xl.sheet_names:
        df = xl.parse(name, **read_excel_kwargs)
        cells = df.shape[0] * df.shape[1]
        if cells > best_cells:
            best_name, best_df, best_cells = name, df, cells
    return best_df


def fetch_cihi_xlsx(url: str, *, sheet: Any = None, timeout: float = 120.0,
                    **read_excel_kwargs: Any):
    """Download a CIHI indicator ``.xlsx`` data table and return a DataFrame.

    Parameters
    ----------
    url : str
        Direct URL of the CIHI ``.xlsx`` data table.
    sheet : int or str, optional
        Worksheet index or name to read.  When omitted (the default),
        the largest sheet is used -- CIHI workbooks lead with a small
        notes sheet, so the largest sheet is the data.  Pass an explicit
        ``sheet`` via the catalog entry's ``fetcher_args`` to override.
    timeout : float
        HTTP timeout in seconds.
    **read_excel_kwargs
        Forwarded to :func:`pandas.read_excel`.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a core dep
        raise ImportError("fetch_cihi_xlsx needs httpx") from exc
    import pandas as pd

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    xl = pd.ExcelFile(io.BytesIO(resp.content))
    if sheet is None:
        return _pick_data_sheet(xl, **read_excel_kwargs)
    return xl.parse(sheet, **read_excel_kwargs)
