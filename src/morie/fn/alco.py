# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Alert co-occurrence rates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def alcooc(
    df: pd.DataFrame,
    *,
    alert_mh_col: str = DEFAULT_COLS["alert_mh"],
    alert_sr_col: str = DEFAULT_COLS["alert_sr"],
    alert_sw_col: str = DEFAULT_COLS["alert_sw"],
) -> pd.DataFrame:
    """Alert co-occurrence rates.

    For each pair of alerts, computes the proportion of rows where
    both alerts are active (= 1). Returns a 3x3 symmetric matrix
    with diagonal = marginal prevalence.

    Parameters
    ----------
    df : DataFrame
        Data with three binary alert columns.
    alert_mh_col : str
        Mental health alert column.
    alert_sr_col : str
        Suicide risk alert column.
    alert_sw_col : str
        Suicide watch alert column.

    Returns
    -------
    DataFrame
        3x3 co-occurrence proportion matrix (labels: mh, sr, sw).
    """
    cols = [alert_mh_col, alert_sr_col, alert_sw_col]
    labels = ["mh", "sr", "sw"]
    data = df[cols].copy()
    for c in cols:
        if pd.api.types.is_string_dtype(data[c]):
            data[c] = (data[c].str.lower() == "yes").astype(int)

    n = len(data)
    if n == 0:
        return pd.DataFrame(np.zeros((3, 3)), index=labels, columns=labels)

    mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            mat[i, j] = float((data[cols[i]] * data[cols[j]]).sum()) / n

    return pd.DataFrame(mat, index=labels, columns=labels)


short = alcooc


def cheatsheet() -> str:
    return "alcooc({}) -> Alert co-occurrence rates."
