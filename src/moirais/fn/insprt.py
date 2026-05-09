# moirais.fn — function file (hadesllm/moirais)
"""Inspection score by facility type."""

from __future__ import annotations

import pandas as pd


def inspection_score(
    df: pd.DataFrame,
    *,
    score_col: str = "Y",
    facility_col: str = "facility_type",
) -> pd.DataFrame:
    """Mean inspection score by facility type.

    Parameters
    ----------
    df : DataFrame
        Records with a numeric score.
    score_col : str
        Numeric score column.
    facility_col : str
        Facility type column.

    Returns
    -------
    DataFrame
        Columns: ``[facility_col, 'mean_score', 'std_score', 'n']``.
    """
    grp = df.groupby(facility_col)[score_col].agg(mean_score="mean", std_score="std", n="count").reset_index()
    return grp


def cheatsheet() -> str:
    return "inspection_score({}) -> Inspection score by facility type."
