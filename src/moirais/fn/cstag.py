# moirais.fn — function file (hadesllm/moirais)
"""Custody age profile over time."""

from __future__ import annotations

import pandas as pd


def custody_age_profile(
    df: pd.DataFrame,
    *,
    age_col: str = "age_group",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Age distribution over fiscal years.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    age_col : str
        Age group column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, age_col, 'count', 'proportion']``.
    """
    counts = df.groupby([year_col, age_col]).size().reset_index(name="count")
    totals = counts.groupby(year_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    return counts.sort_values([year_col, age_col]).reset_index(drop=True)


def cheatsheet() -> str:
    return "custody_age_profile({}) -> Custody age profile over time."
