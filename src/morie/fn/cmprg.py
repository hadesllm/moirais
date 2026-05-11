# morie.fn — function file (hadesllm/morie)
"""Compliance rate by region."""

from __future__ import annotations

import pandas as pd


def compliance_by_region(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    region_col: str = "region",
) -> pd.DataFrame:
    """Compliance rate by region.

    Parameters
    ----------
    df : DataFrame
        Records with a binary compliance flag.
    flag_col : str
        Binary column (1 = compliant).
    region_col : str
        Region column.

    Returns
    -------
    DataFrame
        Columns: ``[region_col, 'n', 'n_compliant', 'rate']``.
    """
    grp = df.groupby(region_col)[flag_col].agg(n="count", n_compliant="sum").reset_index()
    grp["rate"] = grp["n_compliant"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "compliance_by_region({}) -> Compliance rate by region."
