# moirais.fn — function file (hadesllm/moirais)
"""Custody mental health flag trend over years."""

from __future__ import annotations

import pandas as pd


def custody_mental_health(
    df: pd.DataFrame,
    *,
    mh_col: str = "alert_mental_health",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Mental health flag trend by fiscal year.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    mh_col : str
        Binary mental health alert column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'n', 'n_flagged', 'rate']``.
    """
    grp = df.groupby(year_col)[mh_col].agg(n="count", n_flagged="sum").reset_index()
    grp["rate"] = grp["n_flagged"] / grp["n"]
    return grp.sort_values(year_col).reset_index(drop=True)


def cheatsheet() -> str:
    return "custody_mental_health({}) -> Custody mental health flag trend over years."
