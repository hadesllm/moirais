"""Sentence length by age group."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def sentence_by_age(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    age_col: str = DEFAULT_COLS["age"],
) -> pd.DataFrame:
    """Sentence length statistics by age group.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and age columns.
    sent_col : str
        Column with sentence length (days).
    age_col : str
        Column with age group labels.

    Returns
    -------
    DataFrame
        Columns: age_group, mean_sentence, median_sentence, n.
    """
    tmp = df[[age_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(age_col)[sent_col]
        .agg(
            mean_sentence="mean",
            median_sentence="median",
            n="count",
        )
        .reset_index()
    )
    grouped.rename(columns={age_col: "age_group"}, inplace=True)
    return grouped


sntag = sentence_by_age


def cheatsheet() -> str:
    return "sentence_by_age({}) -> Sentence length by age group."
