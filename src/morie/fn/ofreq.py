# morie.fn -- function file (hadesllm/morie)
"""Frequency table with proportions for OTIS correctional data."""

from __future__ import annotations

import pandas as pd


def otis_freq_table(
    df: pd.DataFrame,
    *,
    col: str = "region",
) -> pd.DataFrame:
    """Frequency table with counts, proportions, and cumulative proportions.

    Parameters
    ----------
    df : DataFrame
        Any tabular data.
    col : str
        Column to tabulate.

    Returns
    -------
    DataFrame
        Columns: value, count, proportion, cumulative_proportion.
    """
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = ["value", "count"]
    total = counts["count"].sum()
    counts["proportion"] = (counts["count"] / total).round(4)
    counts["cumulative_proportion"] = counts["proportion"].cumsum().round(4)
    return counts


def cheatsheet() -> str:
    return "otis_freq_table({}) -> Frequency table with proportions for OTIS correctional data."
