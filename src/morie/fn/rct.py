# morie.fn — function file (hadesllm/morie)
"""Restrictive confinement trends over time."""

from __future__ import annotations

import pandas as pd


def rctrnd(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_individual_id",
    year_col: str = "end_fiscal_year",
    region_col: str = "region_at_time_of_placement",
) -> pd.DataFrame:
    """Restrictive confinement trends over time.

    Returns per-year counts by region.

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
        Columns: year, region, n_individuals, n_placements.
    """
    trends = (
        df.groupby([year_col, region_col])
        .agg(
            n_individuals=(id_col, "nunique"),
            n_placements=(id_col, "count"),
        )
        .reset_index()
    )
    trends.columns = ["year", "region", "n_individuals", "n_placements"]
    return trends.sort_values(["year", "region"])


def cheatsheet() -> str:
    return "rctrnd({}) -> Restrictive confinement trends over time."
