# morie.fn -- function file (hadesllm/morie)
"""OTIS cross-tabulation with chi-square test."""

from __future__ import annotations

import pandas as pd
from scipy import stats as sp_stats


def otis_crosstab(
    df: pd.DataFrame,
    *,
    row_col: str = "region",
    col_col: str = "age_group",
) -> dict:
    """Cross-tabulation with chi-square test of independence.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    row_col : str
        Row variable for cross-tab.
    col_col : str
        Column variable for cross-tab.

    Returns
    -------
    dict
        Keys: ``table`` (DataFrame), ``chi2``, ``p_value``, ``dof``,
        ``expected`` (DataFrame).
    """
    table = pd.crosstab(df[row_col], df[col_col])
    chi2, p, dof, expected = sp_stats.chi2_contingency(table)
    return {
        "table": table,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "expected": pd.DataFrame(expected, index=table.index, columns=table.columns),
    }


def cheatsheet() -> str:
    return "otis_crosstab({}) -> OTIS cross-tabulation with chi-square test."
