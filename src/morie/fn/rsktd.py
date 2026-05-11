# morie.fn — function file (hadesllm/morie)
"""Mean risk score trend over years."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def risk_trend(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Mean risk score trend over fiscal years.

    Parameters
    ----------
    df : DataFrame
        Dataset with score and year columns.
    score_col : str
        Column with continuous risk score.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Columns: year, mean_score, sd_score, n.
    """
    tmp = df[[year_col, score_col]].dropna()
    grouped = (
        tmp.groupby(year_col)[score_col]
        .agg(
            mean_score="mean",
            sd_score="std",
            n="count",
        )
        .reset_index()
    )
    grouped = grouped.sort_values(year_col).reset_index(drop=True)
    return grouped


rsktd = risk_trend


def cheatsheet() -> str:
    return "risk_trend({}) -> Mean risk score trend over years."
