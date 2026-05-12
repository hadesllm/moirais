# morie.fn -- function file (hadesllm/morie)
"""Compliance rate by age group."""

from __future__ import annotations

import pandas as pd


def compliance_by_age(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    age_col: str = "age_group",
) -> pd.DataFrame:
    """Compliance rate by age group.

    Parameters
    ----------
    df : DataFrame
        Records with a binary compliance flag.
    flag_col : str
        Binary column (1 = compliant).
    age_col : str
        Age group column.

    Returns
    -------
    DataFrame
        Columns: ``[age_col, 'n', 'n_compliant', 'rate']``.
    """
    grp = df.groupby(age_col)[flag_col].agg(n="count", n_compliant="sum").reset_index()
    grp["rate"] = grp["n_compliant"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "compliance_by_age({}) -> Compliance rate by age group."
