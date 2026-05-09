"""Volatility by origin region."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def vol_reg(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    region_col: str = DEFAULT_COLS["region"],
) -> pd.DataFrame:
    """Volatility stratified by initial (origin) region.

    For each individual, volatility = number of distinct regions observed
    across all years. Results are grouped by the individual's first
    observed region.

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

    Returns
    -------
    DataFrame
        Columns: origin_region, mean_volatility, median_volatility, n.
    """
    data = df.sort_values([id_col, year_col])

    # First region per individual
    first_region = data.groupby(id_col)[region_col].first().rename("origin_region")

    # Volatility = distinct regions per individual
    vol = data.groupby(id_col)[region_col].nunique().rename("volatility")

    merged = pd.concat([first_region, vol], axis=1).reset_index()

    result = (
        merged.groupby("origin_region")
        .agg(
            mean_volatility=("volatility", "mean"),
            median_volatility=("volatility", "median"),
            n=("volatility", "count"),
        )
        .reset_index()
    )
    return result


short = vol_reg


def cheatsheet() -> str:
    return "vol_reg({}) -> Volatility by origin region."
