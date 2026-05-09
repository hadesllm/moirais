"""Volatility trend over time."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def vol_trd(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    region_col: str = DEFAULT_COLS["region"],
) -> pd.DataFrame:
    """Mean volatility trend over time.

    For each fiscal year, computes the mean number of distinct regions
    each individual has been observed in up to and including that year.

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
        Columns: year, mean_volatility, n_individuals.
    """
    data = df.sort_values([id_col, year_col])
    years = sorted(data[year_col].dropna().unique())

    rows = []
    for yr in years:
        subset = data[data[year_col] <= yr]
        vol = subset.groupby(id_col)[region_col].nunique()
        rows.append(
            {
                "year": yr,
                "mean_volatility": float(vol.mean()),
                "n_individuals": int(vol.count()),
            }
        )

    return pd.DataFrame(rows)


short = vol_trd


def cheatsheet() -> str:
    return "vol_trd({}) -> Volatility trend over time."
