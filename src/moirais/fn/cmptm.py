# moirais.fn — function file (hadesllm/moirais)
"""Compliance rate over time."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def compliance_timeline(
    df: pd.DataFrame,
    *,
    compliant_col: str = "compliant",
    period_col: str = "period",
) -> DescriptiveResult:
    """Compliance proportion per time period.

    Parameters
    ----------
    df : DataFrame
    compliant_col : str
        Binary compliance indicator.
    period_col : str

    Returns
    -------
    DescriptiveResult
    """
    grouped = df.groupby(period_col)[compliant_col].agg(["mean", "count"]).reset_index()
    grouped.columns = ["period", "rate", "n"]
    return DescriptiveResult(name="compliance_timeline", value=grouped)


cmptm = compliance_timeline


def cheatsheet() -> str:
    return "compliance_timeline({}) -> Compliance rate over time."
