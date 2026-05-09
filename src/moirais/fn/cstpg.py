# moirais.fn — function file (hadesllm/moirais)
"""Custody program participation rate by region."""

from __future__ import annotations

import pandas as pd


def custody_program_rate(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    group_col: str = "region",
) -> pd.DataFrame:
    """Program participation rate by group.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    flag_col : str
        Binary flag column (1 = participated).
    group_col : str
        Grouping column (e.g. region).

    Returns
    -------
    DataFrame
        Columns: ``[group_col, 'n', 'n_participated', 'rate']``.
    """
    grp = df.groupby(group_col)[flag_col].agg(n="count", n_participated="sum").reset_index()
    grp["rate"] = grp["n_participated"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "custody_program_rate({}) -> Custody program participation rate by region."
