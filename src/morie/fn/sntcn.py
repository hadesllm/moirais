"""Concurrent vs consecutive sentence analysis."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def sentence_concurrency(
    df: pd.DataFrame,
    *,
    type_col: str = "sentence_type",
    sentence_col: str = "sentence_days",
) -> DescriptiveResult:
    """Concurrent vs consecutive sentence analysis.

    Parameters
    ----------
    df : DataFrame
    type_col : str
        Column with 'concurrent' or 'consecutive' labels.
    sentence_col : str

    Returns
    -------
    DescriptiveResult
    """
    grouped = df.groupby(type_col)[sentence_col].agg(["mean", "median", "count"]).reset_index()
    grouped.columns = ["type", "mean", "median", "n"]
    return DescriptiveResult(name="sentence_concurrency", value=grouped)


sntcn = sentence_concurrency


def cheatsheet() -> str:
    return "sentence_concurrency({}) -> Concurrent vs consecutive sentence analysis."
