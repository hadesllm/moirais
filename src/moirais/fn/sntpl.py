"""Sentence length by plea type."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def sentence_plea(
    df: pd.DataFrame,
    *,
    plea_col: str = "plea_type",
    sentence_col: str = "sentence_days",
) -> DescriptiveResult:
    """Sentence length by plea type (guilty/trial).

    Parameters
    ----------
    df : DataFrame
    plea_col : str
    sentence_col : str

    Returns
    -------
    DescriptiveResult
    """
    grouped = df.groupby(plea_col)[sentence_col].agg(["mean", "median", "std", "count"]).reset_index()
    grouped.columns = ["plea", "mean", "median", "std", "n"]
    return DescriptiveResult(name="sentence_plea", value=grouped)


sntpl = sentence_plea


def cheatsheet() -> str:
    return "sentence_plea({}) -> Sentence length by plea type."
