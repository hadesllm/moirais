# moirais.fn — function file (hadesllm/moirais)
"""Region placement trend over years."""

from __future__ import annotations

import pandas as pd

from ._otis_const import DEFAULT_COLS


def rplace_region_trend(
    df: pd.DataFrame,
    *,
    id_col: str = DEFAULT_COLS["id"],
    region_col: str = DEFAULT_COLS["region"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Count unique individuals per (year, region) combination.

    Parameters
    ----------
    df : DataFrame
        Correctional placement data.
    id_col : str
        Column with unique individual identifiers.
    region_col : str
        Column with region labels.
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Three columns: ``year``, ``region``, ``n_individuals``.
    """
    counts = df.groupby([year_col, region_col])[id_col].nunique().reset_index()
    counts.columns = ["year", "region", "n_individuals"]
    return counts.sort_values(["year", "region"]).reset_index(drop=True)


rpl_rt = rplace_region_trend


def cheatsheet() -> str:
    return "rplace_region_trend({}) -> Region placement trend over years."
