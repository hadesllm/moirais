# SPDX-License-Identifier: AGPL-3.0-or-later
"""Statistics Canada direct-download ingest.

Statistics Canada distributes Public Use Microdata Files (PUMFs) and
other products from ``www150.statcan.gc.ca/n1/pub/...`` as ``.zip``
archives containing one or more CSV files.  Note that a StatCan
*catalogue* page (e.g. ``/n1/en/catalogue/82M0013X``) is only an HTML
index — the actual data is linked from the *product* page
(``/n1/pub/82m0013x/82m0013x2024001-eng.htm``), which points at the
real ``..._CSV.zip``.

:func:`fetch_statcan_csv` downloads such a zip and returns its CSV as a
:class:`pandas.DataFrame`.  It is wired into
:data:`morie.data.DATASET_CATALOG` through the ``fetcher`` mechanism,
so ``morie.load_dataset()`` can fetch a StatCan product on demand.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from typing import Any

__all__ = ["fetch_statcan_csv"]


def _csv_from_zip(zip_path: str, member: str | None = None,
                  **read_csv_kwargs: Any):
    """Read a CSV member out of a zip archive into a DataFrame.

    If ``member`` is omitted the first ``.csv`` entry is used.
    """
    import pandas as pd

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError("no .csv file inside the StatCan archive")
        chosen = member or csvs[0]
        if chosen not in names:
            raise KeyError(
                f"member {chosen!r} not in the archive; CSVs present: {csvs}"
            )
        kwargs = {"low_memory": False, **read_csv_kwargs}
        with zf.open(chosen) as fh:
            return pd.read_csv(fh, **kwargs)


def fetch_statcan_csv(
    url: str,
    *,
    member: str | None = None,
    timeout: float = 600.0,
    **read_csv_kwargs: Any,
):
    """Download a StatCan ``..._CSV.zip`` product and return a DataFrame.

    The archive is streamed to a temporary file (StatCan PUMF zips can
    be hundreds of megabytes), the CSV is extracted, and the temporary
    file is removed.

    Parameters
    ----------
    url : str
        Direct URL of the StatCan ``.zip`` product, e.g.
        ``https://www150.statcan.gc.ca/n1/pub/82m0013x/2024001/2022_CSV.zip``.
    member : str, optional
        Name of the CSV inside the archive; defaults to the first
        ``.csv`` entry.
    timeout : float
        HTTP timeout in seconds.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a core dep
        raise ImportError("fetch_statcan_csv needs httpx") from exc

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    tmp.write(chunk)
        tmp.close()
        return _csv_from_zip(tmp.name, member, **read_csv_kwargs)
    finally:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
