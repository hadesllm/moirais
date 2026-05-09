# moirais.fn — function file (hadesllm/moirais)
"""Age group x gender cross-tabulation of placements."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_age_gender(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    age_col: str = DEFAULT_COLS["age"],
    gender_col: str = DEFAULT_COLS["gender"],
) -> pd.DataFrame:
    """Cross-tabulate unique individual counts by age group and gender.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    age_col : str
        Column with age group labels.
    gender_col : str
        Column with gender labels.

    Returns
    -------
    DataFrame
        Cross-tab with age groups as rows, genders as columns.
    """
    deduped = df.drop_duplicates(subset=[id_col, age_col, gender_col])
    return pd.crosstab(deduped[age_col], deduped[gender_col], margins=True)


rpl_ag = rplace_age_gender


def cheatsheet() -> str:
    return "rplace_age_gender({}) -> Age group x gender cross-tabulation of placements."
