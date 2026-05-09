# moirais.fn — function file (hadesllm/moirais)
"""Recidivism rate by gender."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def recidivism_by_gender(
    df: pd.DataFrame,
    *,
    outcome: str = DEFAULT_COLS["outcome"],
    gender_col: str = DEFAULT_COLS["gender"],
) -> pd.DataFrame:
    """Recidivism rate by gender.

    Parameters
    ----------
    df : DataFrame
        Dataset with outcome and gender columns.
    outcome : str
        Column with recidivism outcome (binary: >0 = recid).
    gender_col : str
        Column with gender labels.

    Returns
    -------
    DataFrame
        Columns: gender, n_total, n_recid, rate.
    """
    tmp = df[[gender_col, outcome]].dropna()
    tmp = tmp.assign(_recid=(tmp[outcome] > 0).astype(int))
    grouped = (
        tmp.groupby(gender_col)
        .agg(
            n_total=("_recid", "count"),
            n_recid=("_recid", "sum"),
        )
        .reset_index()
    )
    grouped.columns = ["gender", "n_total", "n_recid"]
    grouped["rate"] = grouped["n_recid"] / grouped["n_total"]
    return grouped


rcd_g = recidivism_by_gender


def cheatsheet() -> str:
    return "recidivism_by_gender({}) -> Recidivism rate by gender."
