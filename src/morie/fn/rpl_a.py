# morie.fn -- function file (hadesllm/morie)
"""Placement counts by age group."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_by_age(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    age_col: str = DEFAULT_COLS["age"],
) -> pd.DataFrame:
    """Count unique individuals per age group.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    age_col : str
        Column with age group labels.

    Returns
    -------
    DataFrame
        Two columns: ``age_group`` and ``n_individuals``.
    """
    counts = df.groupby(age_col)[id_col].nunique().reset_index()
    counts.columns = ["age_group", "n_individuals"]
    return counts.sort_values("n_individuals", ascending=False).reset_index(drop=True)


rpl_a = rplace_by_age


def cheatsheet() -> str:
    return "rplace_by_age({}) -> Placement counts by age group."
