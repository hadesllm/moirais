# morie.fn — function file (hadesllm/morie)
"""Custody occupancy count per facility per year."""

from __future__ import annotations

import pandas as pd


def custody_occupancy(
    df: pd.DataFrame,
    *,
    year_col: str = "end_fiscal_year",
    facility_col: str = "facility_type",
) -> pd.DataFrame:
    """Count of records per facility type per year.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    year_col : str
        Fiscal year column.
    facility_col : str
        Facility type column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, facility_col, 'count']``.
    """
    out = (
        df.groupby([year_col, facility_col])
        .size()
        .reset_index(name="count")
        .sort_values([year_col, facility_col])
        .reset_index(drop=True)
    )
    return out


def cheatsheet() -> str:
    return "custody_occupancy({}) -> Custody occupancy count per facility per year."
