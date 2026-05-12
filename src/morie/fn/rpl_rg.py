# morie.fn -- function file (hadesllm/morie)
"""Region x gender cross-tabulation of placements."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_region_gender(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    region_col: str = DEFAULT_COLS["region"],
    gender_col: str = DEFAULT_COLS["gender"],
) -> pd.DataFrame:
    """Cross-tabulate unique individual counts by region and gender.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    region_col : str
        Column with region labels.
    gender_col : str
        Column with gender labels.

    Returns
    -------
    DataFrame
        Cross-tab with regions as rows, genders as columns.
    """
    deduped = df.drop_duplicates(subset=[id_col, region_col, gender_col])
    return pd.crosstab(deduped[region_col], deduped[gender_col], margins=True)


rpl_rg = rplace_region_gender


def cheatsheet() -> str:
    return "rplace_region_gender({}) -> Region x gender cross-tabulation of placements."
