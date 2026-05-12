# morie.fn -- function file (hadesllm/morie)
"""Custody grievance rate by facility type."""

from __future__ import annotations

import pandas as pd


def custody_grievance_rate(
    df: pd.DataFrame,
    *,
    event_col: str = "D",
    group_col: str = "facility_type",
) -> pd.DataFrame:
    """Event rate by facility type (grievance proxy).

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    event_col : str
        Binary event column (1 = event).
    group_col : str
        Grouping column (e.g. facility_type).

    Returns
    -------
    DataFrame
        Columns: ``[group_col, 'n', 'n_events', 'rate']``.
    """
    grp = df.groupby(group_col)[event_col].agg(n="count", n_events="sum").reset_index()
    grp["rate"] = grp["n_events"] / grp["n"]
    return grp


def cheatsheet() -> str:
    return "custody_grievance_rate({}) -> Custody grievance rate by facility type."
