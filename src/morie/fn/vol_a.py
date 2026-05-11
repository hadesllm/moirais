"""Volatility by age group."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def vol_age(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    region_col: str = DEFAULT_COLS["region"],
    age_col: str = DEFAULT_COLS["age"],
) -> pd.DataFrame:
    """Volatility stratified by age group.

    For each individual, volatility = number of distinct regions observed
    across all years. Results are grouped by the individual's first
    observed age group.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Unique individual identifier column.
    year_col : str
        Fiscal year column.
    region_col : str
        Region column.
    age_col : str
        Age group column.

    Returns
    -------
    DataFrame
        Columns: age_group, mean_volatility, median_volatility, n.
    """
    data = df.sort_values([id_col, year_col])

    first_age = data.groupby(id_col)[age_col].first().rename("age_group")
    vol = data.groupby(id_col)[region_col].nunique().rename("volatility")

    merged = pd.concat([first_age, vol], axis=1).reset_index()

    result = (
        merged.groupby("age_group")
        .agg(
            mean_volatility=("volatility", "mean"),
            median_volatility=("volatility", "median"),
            n=("volatility", "count"),
        )
        .reset_index()
    )
    return result


short = vol_age


def cheatsheet() -> str:
    return "vol_age({}) -> Volatility by age group."
