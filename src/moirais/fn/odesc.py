# moirais.fn — function file (hadesllm/moirais)
"""OTIS demographic summary — region x age x gender counts."""

from __future__ import annotations

import pandas as pd


def otis_demographic_summary(
    df: pd.DataFrame,
    *,
    region_col: str = "region",
    age_col: str = "age_group",
    gender_col: str = "gender",
) -> pd.DataFrame:
    """Full demographic breakdown (region x age x gender counts).

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    region_col : str
        Region column.
    age_col : str
        Age group column.
    gender_col : str
        Gender column.

    Returns
    -------
    DataFrame
        Columns: ``[region_col, age_col, gender_col, 'count']``.
    """
    out = (
        df.groupby([region_col, age_col, gender_col])
        .size()
        .reset_index(name="count")
        .sort_values([region_col, age_col, gender_col])
        .reset_index(drop=True)
    )
    return out


def cheatsheet() -> str:
    return "otis_demographic_summary({}) -> OTIS demographic summary — region x age x gender counts."
