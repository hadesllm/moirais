# moirais.fn — function file (hadesllm/moirais)
"""Custody health access — alert rate by group."""

from __future__ import annotations

import pandas as pd


def custody_health_access(
    df: pd.DataFrame,
    *,
    alert_col: str = "alert_mental_health",
    group_col: str = "region",
) -> pd.DataFrame:
    """Health alert rate by group.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    alert_col : str
        Binary alert column (1 = flagged).
    group_col : str
        Grouping column (e.g. region, facility).

    Returns
    -------
    DataFrame
        Columns: ``[group_col, 'n', 'n_flagged', 'rate']``.
    """
    grp = df.groupby(group_col)[alert_col].agg(n="count", n_flagged="sum").reset_index()
    grp["rate"] = grp["n_flagged"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "custody_health_access({}) -> Custody health access — alert rate by group."
