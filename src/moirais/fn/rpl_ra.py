# moirais.fn — function file (hadesllm/moirais)
"""Region x age group cross-tabulation of placements."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_region_age(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    region_col: str = DEFAULT_COLS["region"],
    age_col: str = DEFAULT_COLS["age"],
) -> pd.DataFrame:
    """Cross-tabulate unique individual counts by region and age group.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    region_col : str
        Column with region labels.
    age_col : str
        Column with age group labels.

    Returns
    -------
    DataFrame
        Cross-tab with regions as rows, age groups as columns.
    """
    deduped = df.drop_duplicates(subset=[id_col, region_col, age_col])
    return pd.crosstab(deduped[region_col], deduped[age_col], margins=True)


rpl_ra = rplace_region_age


def cheatsheet() -> str:
    return "rplace_region_age({}) -> Region x age group cross-tabulation of placements."
