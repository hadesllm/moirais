# morie.fn -- function file (hadesllm/morie)
"""Placement trend over time (year-level counts)."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_trend(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Count unique individuals placed per year.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Two columns: ``year`` and ``n_individuals``, sorted by year.
    """
    counts = df.groupby(year_col)[id_col].nunique().reset_index()
    counts.columns = ["year", "n_individuals"]
    return counts.sort_values("year").reset_index(drop=True)


rpl_t = rplace_trend


def cheatsheet() -> str:
    return "rplace_trend({}) -> Placement trend over time (year-level counts)."
