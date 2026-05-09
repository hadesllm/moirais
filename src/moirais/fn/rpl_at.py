# moirais.fn — function file (hadesllm/moirais)
"""Age group placement trend over years."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_age_trend(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    age_col: str = DEFAULT_COLS["age"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Count unique individuals per (year, age group) combination.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    age_col : str
        Column with age group labels.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Three columns: ``year``, ``age_group``, ``n_individuals``.
    """
    counts = df.groupby([year_col, age_col])[id_col].nunique().reset_index()
    counts.columns = ["year", "age_group", "n_individuals"]
    return counts.sort_values(["year", "age_group"]).reset_index(drop=True)


rpl_at = rplace_age_trend


def cheatsheet() -> str:
    return "rplace_age_trend({}) -> Age group placement trend over years."
