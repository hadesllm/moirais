# morie.fn -- function file (hadesllm/morie)
"""Compliance trend over fiscal years."""

from __future__ import annotations

import pandas as pd


def compliance_trend(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Compliance rate trend by fiscal year.

    Parameters
    ----------
    df : DataFrame
        Records with a binary compliance flag.
    flag_col : str
        Binary column (1 = compliant).
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'n', 'n_compliant', 'rate']``.
    """
    grp = df.groupby(year_col)[flag_col].agg(n="count", n_compliant="sum").reset_index()
    grp["rate"] = grp["n_compliant"] / grp["n"]
    return grp.sort_values(year_col).reset_index(drop=True)


def cheatsheet() -> str:
    return "compliance_trend({}) -> Compliance trend over fiscal years."
