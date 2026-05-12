# morie.fn -- function file (hadesllm/morie)
"""Demographic trend over time."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def otis_demo_trend(
    df: pd.DataFrame,
    *,
    demo_col: str = "group",
    period_col: str = "period",
) -> DescriptiveResult:
    """Demographic composition trend over periods.

    Parameters
    ----------
    df : DataFrame
    demo_col : str
    period_col : str

    Returns
    -------
    DescriptiveResult
    """
    ct = df.groupby([period_col, demo_col]).size().reset_index(name="count")
    totals = df.groupby(period_col).size().reset_index(name="total")
    ct = ct.merge(totals, on=period_col)
    ct["proportion"] = ct["count"] / ct["total"]
    pivot = ct.pivot_table(index=period_col, columns=demo_col, values="proportion", fill_value=0)
    return DescriptiveResult(name="otis_demo_trend", value=pivot)


odm_t = otis_demo_trend


def cheatsheet() -> str:
    return "otis_demo_trend({}) -> Demographic trend over time."
