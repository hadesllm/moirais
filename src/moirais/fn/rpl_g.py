# moirais.fn — function file (hadesllm/moirais)
"""Placement counts by gender."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_by_gender(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    gender_col: str = DEFAULT_COLS["gender"],
) -> pd.DataFrame:
    """Count unique individuals per gender category.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    gender_col : str
        Column with gender labels.

    Returns
    -------
    DataFrame
        Two columns: ``gender`` and ``n_individuals``.
    """
    counts = df.groupby(gender_col)[id_col].nunique().reset_index()
    counts.columns = ["gender", "n_individuals"]
    return counts.sort_values("n_individuals", ascending=False).reset_index(drop=True)


rpl_g = rplace_by_gender


def cheatsheet() -> str:
    return "rplace_by_gender({}) -> Placement counts by gender."
