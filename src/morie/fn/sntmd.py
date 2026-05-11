"""Median sentence length by group."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def sentence_by_group(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    group_col: str = DEFAULT_COLS["region"],
) -> pd.DataFrame:
    """Median sentence length by group.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and group columns.
    sent_col : str
        Column with sentence length (days).
    group_col : str
        Column with group labels.

    Returns
    -------
    DataFrame
        Columns: group, median_sentence, mean_sentence, n.
    """
    tmp = df[[group_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(group_col)[sent_col]
        .agg(
            median_sentence="median",
            mean_sentence="mean",
            n="count",
        )
        .reset_index()
    )
    grouped.rename(columns={group_col: "group"}, inplace=True)
    return grouped


sntmd = sentence_by_group


def cheatsheet() -> str:
    return "sentence_by_group({}) -> Median sentence length by group."
