# morie.fn -- function file (hadesllm/morie)
"""Gender placement trend over years."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_gender_trend(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    gender_col: str = DEFAULT_COLS["gender"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Count unique individuals per (year, gender) combination.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    gender_col : str
        Column with gender labels.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Three columns: ``year``, ``gender``, ``n_individuals``.
    """
    counts = df.groupby([year_col, gender_col])[id_col].nunique().reset_index()
    counts.columns = ["year", "gender", "n_individuals"]
    return counts.sort_values(["year", "gender"]).reset_index(drop=True)


rpl_gt = rplace_gender_trend


def cheatsheet() -> str:
    return "rplace_gender_trend({}) -> Gender placement trend over years."
