# morie.fn -- function file (hadesllm/morie)
"""Chi-squared test of independence for OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def otis_chi2_test(
    df: pd.DataFrame,
    *,
    row_col: str = "region",
    col_col: str = "gender",
) -> dict:
    """Chi-squared test of independence between two categorical columns.

    Parameters
    ----------
    df : DataFrame
        Data with two categorical columns.
    row_col, col_col : str
        Column names for rows and columns of the contingency table.

    Returns
    -------
    dict
        Keys: chi2, pval, df, n, cramers_v, contingency_table.
    """
    ct = pd.crosstab(df[row_col], df[col_col])
    chi2, pval, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0

    return {
        "chi2": float(chi2),
        "pval": float(pval),
        "df": int(dof),
        "n": int(n),
        "cramers_v": round(v, 4),
        "contingency_table": ct,
    }


def cheatsheet() -> str:
    return "otis_chi2_test({}) -> Chi-squared test of independence for OTIS correctional data."
