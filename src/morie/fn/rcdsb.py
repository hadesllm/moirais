# morie.fn -- function file (hadesllm/morie)
"""Recidivism rate by subgroup."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def recidivism_subgroup(
    df: pd.DataFrame,
    *,
    recid_col: str = "recidivism",
    group_col: str = "group",
) -> DescriptiveResult:
    """Recidivism rate by subgroup (region, age, gender, etc.).

    Parameters
    ----------
    df : DataFrame
        Data with recidivism indicator and group column.
    recid_col : str
        Binary recidivism column.
    group_col : str
        Grouping column.

    Returns
    -------
    DescriptiveResult
        value is DataFrame with group, rate, n, n_recid.
    """
    grouped = df.groupby(group_col)[recid_col].agg(["mean", "sum", "count"]).reset_index()
    grouped.columns = ["group", "rate", "n_recid", "n"]
    grouped["n_recid"] = grouped["n_recid"].astype(int)
    return DescriptiveResult(
        name="recidivism_subgroup",
        value=grouped,
        extra={"n_groups": len(grouped), "overall_rate": float(df[recid_col].mean())},
    )


rcdsb = recidivism_subgroup


def cheatsheet() -> str:
    return "recidivism_subgroup({}) -> Recidivism rate by subgroup."
