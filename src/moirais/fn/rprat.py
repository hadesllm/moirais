# moirais.fn — function file (hadesllm/moirais)
"""Placement rate per 100,000 population."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_rate(
    df: pd.DataFrame,
    population: Union[int, float, str] = 100_000,
    *,
    id_col: str = DEFAULT_COLS["id"],
    year_col: str = DEFAULT_COLS["year"],
    pop_col: str = "population",
    per: int = 100_000,
) -> pd.DataFrame:
    """Compute placement rate per ``per`` population by year.

    If *population* is numeric, it is used as a constant denominator for
    every year.  If it is a string, it names a column in *df* that
    contains year-specific population counts.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    population : int, float, or str
        Constant population value **or** column name holding population.
    id_col : str
        Column with unique individual identifiers.
    year_col : str
        Column with fiscal year.
    pop_col : str
        Ignored unless *population* is the default string ``"population"``.
    per : int
        Denominator scaling (default 100,000).

    Returns
    -------
    DataFrame
        Columns: ``year``, ``n_individuals``, ``population``, ``rate_per_100k``.
    """
    counts = df.groupby(year_col)[id_col].nunique().reset_index()
    counts.columns = ["year", "n_individuals"]

    if isinstance(population, str):
        # Merge population from a column already in df (one row per year)
        pop_df = df.groupby(year_col)[population].first().reset_index()
        pop_df.columns = ["year", "population"]
        counts = counts.merge(pop_df, on="year", how="left")
    else:
        counts["population"] = float(population)

    counts["rate_per_100k"] = np.where(
        counts["population"] > 0,
        counts["n_individuals"] / counts["population"] * per,
        np.nan,
    )
    return counts.sort_values("year").reset_index(drop=True)


rprat = rplace_rate


def cheatsheet() -> str:
    return "rplace_rate({}) -> Placement rate per 100,000 population."
