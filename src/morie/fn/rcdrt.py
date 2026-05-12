# morie.fn -- function file (hadesllm/morie)
"""Recidivism rate trend over years."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def recidivism_trend(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Recidivism rate trend over years.

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome and year columns.
    outcome : str
        Column with recidivism outcome (binary: >0 = recid).
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Columns: year, n_total, n_recid, rate.
    """
    tmp = df[[year_col, outcome]].dropna()
    tmp = tmp.assign(_recid=(tmp[outcome] > 0).astype(int))
    grouped = (
        tmp.groupby(year_col)
        .agg(
            n_total=("_recid", "count"),
            n_recid=("_recid", "sum"),
        )
        .reset_index()
    )
    grouped.columns = ["year", "n_total", "n_recid"]
    grouped["rate"] = grouped["n_recid"] / grouped["n_total"]
    grouped = grouped.sort_values("year").reset_index(drop=True)
    return grouped


rcdrt = recidivism_trend


def cheatsheet() -> str:
    return "recidivism_trend({}) -> Recidivism rate trend over years."
