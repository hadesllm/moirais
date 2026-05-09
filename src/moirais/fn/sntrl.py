"""Sentence length trends over years."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def sentence_by_year(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    year_col: str = DEFAULT_COLS["year"],
) -> pd.DataFrame:
    """Mean and median sentence length over fiscal years.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and year columns.
    sent_col : str
        Column with sentence length (days).
    year_col : str
        Column with fiscal year.

    Returns
    -------
    DataFrame
        Columns: year, mean_sentence, median_sentence, n.
    """
    tmp = df[[year_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(year_col)[sent_col]
        .agg(
            mean_sentence="mean",
            median_sentence="median",
            n="count",
        )
        .reset_index()
    )
    grouped = grouped.sort_values(year_col).reset_index(drop=True)
    return grouped


sntrl = sentence_by_year


def cheatsheet() -> str:
    return "sentence_by_year({}) -> Sentence length trends over years."
