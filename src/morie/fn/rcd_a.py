# morie.fn — function file (hadesllm/morie)
"""Recidivism rate by age group."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def recidivism_by_age(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    age_col: str = DEFAULT_COLS["age"],
) -> pd.DataFrame:
    """Recidivism rate by age group.

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome and age columns.
    outcome : str
        Column with recidivism outcome (binary: >0 = recid).
    age_col : str
        Column with age group labels.

    Returns
    -------
    DataFrame
        Columns: age_group, n_total, n_recid, rate.
    """
    tmp = df[[age_col, outcome]].dropna()
    tmp = tmp.assign(_recid=(tmp[outcome] > 0).astype(int))
    grouped = (
        tmp.groupby(age_col)
        .agg(
            n_total=("_recid", "count"),
            n_recid=("_recid", "sum"),
        )
        .reset_index()
    )
    grouped.columns = ["age_group", "n_total", "n_recid"]
    grouped["rate"] = grouped["n_recid"] / grouped["n_total"]
    return grouped


rcd_a = recidivism_by_age


def cheatsheet() -> str:
    return "recidivism_by_age({}) -> Recidivism rate by age group."
