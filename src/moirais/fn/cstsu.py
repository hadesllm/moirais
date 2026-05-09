# moirais.fn — function file (hadesllm/moirais)
"""Custody substance flag by age group."""

from __future__ import annotations

import pandas as pd


def custody_substance(
    df: pd.DataFrame,
    *,
    flag_col: str = "D",
    group_col: str = "age_group",
) -> pd.DataFrame:
    """Substance-related flag rate by group.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    flag_col : str
        Binary flag column (1 = positive).
    group_col : str
        Grouping column (e.g. age_group).

    Returns
    -------
    DataFrame
        Columns: ``[group_col, 'n', 'n_flagged', 'rate']``.
    """
    grp = df.groupby(group_col)[flag_col].agg(n="count", n_flagged="sum").reset_index()
    grp["rate"] = grp["n_flagged"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "custody_substance({}) -> Custody substance flag by age group."
