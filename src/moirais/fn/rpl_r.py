# moirais.fn — function file (hadesllm/moirais)
"""Placement counts filtered to a single region."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_by_region(
    df: pd.DataFrame,
    region: str,
    *,
    id_col: str = DEFAULT_COLS["id"],
    region_col: str = DEFAULT_COLS["region"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Filter placements to one region and return per-year counts.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    region : str
        Region value to filter on.
    id_col : str
        Column with unique individual identifiers.
    region_col : str
        Column with region labels.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Two columns: ``year`` and ``n_individuals``.
    """
    sub = df.loc[df[region_col] == region]
    counts = sub.groupby(year_col)[id_col].nunique().reset_index()
    counts.columns = ["year", "n_individuals"]
    return counts


rpl_r = rplace_by_region


def cheatsheet() -> str:
    return "rplace_by_region({}) -> Placement counts filtered to a single region."
