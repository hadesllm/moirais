# morie.fn — function file (hadesllm/morie)
"""OTIS trend summary — year-over-year change statistics."""

from __future__ import annotations

import pandas as pd


def otis_trend_summary(
    df: pd.DataFrame,
    *,
    metric_col: str = "Y",
    year_col: str = "end_fiscal_year",
) -> pd.DataFrame:
    """Year-over-year mean and change statistics for a metric.

    Parameters
    ----------
    df : DataFrame
        Correctional records.
    metric_col : str
        Numeric metric column.
    year_col : str
        Fiscal year column.

    Returns
    -------
    DataFrame
        Columns: ``[year_col, 'mean', 'std', 'n', 'change', 'pct_change']``.
    """
    grp = (
        df.groupby(year_col)[metric_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(year_col)
        .reset_index(drop=True)
    )
    grp["change"] = grp["mean"].diff()
    grp["pct_change"] = grp["mean"].pct_change() * 100
    return grp


def cheatsheet() -> str:
    return "otis_trend_summary({}) -> OTIS trend summary — year-over-year change statistics."
