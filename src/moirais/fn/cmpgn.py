# moirais.fn — function file (hadesllm/moirais)
"""Compliance rate by gender."""

from __future__ import annotations

import pandas as pd


def compliance_by_gender(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    gender_col: str = "gender",
) -> pd.DataFrame:
    """Compliance rate by gender.

    Parameters
    ----------
    df : DataFrame
        Records with a binary compliance flag.
    flag_col : str
        Binary column (1 = compliant).
    gender_col : str
        Gender column.

    Returns
    -------
    DataFrame
        Columns: ``[gender_col, 'n', 'n_compliant', 'rate']``.
    """
    grp = df.groupby(gender_col)[flag_col].agg(n="count", n_compliant="sum").reset_index()
    grp["rate"] = grp["n_compliant"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "compliance_by_gender({}) -> Compliance rate by gender."
