"""Sentence length by gender."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def sentence_by_gender(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    gender_col: str = DEFAULT_COLS["gender"],
) -> pd.DataFrame:
    """Sentence length statistics by gender.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and gender columns.
    sent_col : str
        Column with sentence length (days).
    gender_col : str
        Column with gender labels.

    Returns
    -------
    DataFrame
        Columns: gender, mean_sentence, median_sentence, n.
    """
    tmp = df[[gender_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(gender_col)[sent_col]
        .agg(
            mean_sentence="mean",
            median_sentence="median",
            n="count",
        )
        .reset_index()
    )
    grouped.rename(columns={gender_col: "gender"}, inplace=True)
    return grouped


sntgn = sentence_by_gender


def cheatsheet() -> str:
    return "sentence_by_gender({}) -> Sentence length by gender."
