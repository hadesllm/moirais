# moirais.fn — function file (hadesllm/moirais)
"""Inspection score trend over fiscal years."""

from __future__ import annotations

import pandas as pd


def inspection_trend(
    df: pd.DataFrame,
    *,
    score_col: str = "Y",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Mean inspection score trend by fiscal year.

    Parameters
    ----------
    df : DataFrame
        Records with a numeric score.
    score_col : str
        Numeric score column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'mean_score', 'std_score', 'n']``.
    """
    grp = df.groupby(year_col)[score_col].agg(mean_score="mean", std_score="std", n="count").reset_index()
    return grp.sort_values(year_col).reset_index(drop=True)


def cheatsheet() -> str:
    return "inspection_trend({}) -> Inspection score trend over fiscal years."
