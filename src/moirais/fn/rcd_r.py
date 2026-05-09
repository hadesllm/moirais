# moirais.fn — function file (hadesllm/moirais)
"""Recidivism rate by region."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def recidivism_by_region(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    region_col: str = DEFAULT_COLS["region"],
) -> pd.DataFrame:
    """Recidivism rate by region.

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome and region columns.
    outcome : str
        Column with recidivism outcome (binary: >0 = recid).
    region_col : str
        Column with region labels.

    Returns
    -------
    DataFrame
        Columns: region, n_total, n_recid, rate.
    """
    tmp = df[[region_col, outcome]].dropna()
    tmp = tmp.assign(_recid=(tmp[outcome] > 0).astype(int))
    grouped = (
        tmp.groupby(region_col)
        .agg(
            n_total=("_recid", "count"),
            n_recid=("_recid", "sum"),
        )
        .reset_index()
    )
    grouped.columns = ["region", "n_total", "n_recid"]
    grouped["rate"] = grouped["n_recid"] / grouped["n_total"]
    return grouped


rcd_r = recidivism_by_region


def cheatsheet() -> str:
    return "recidivism_by_region({}) -> Recidivism rate by region."
