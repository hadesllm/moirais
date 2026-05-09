"""Sentence length by region."""

from __future__ import annotations

import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def sentence_by_region(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    region_col: str = DEFAULT_COLS["region"],
) -> pd.DataFrame:
    """Sentence length statistics by region.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and region columns.
    sent_col : str
        Column with sentence length (days).
    region_col : str
        Column with region labels.

    Returns
    -------
    DataFrame
        Columns: region, mean_sentence, median_sentence, n.
    """
    tmp = df[[region_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(region_col)[sent_col]
        .agg(
            mean_sentence="mean",
            median_sentence="median",
            n="count",
        )
        .reset_index()
    )
    grouped.rename(columns={region_col: "region"}, inplace=True)
    return grouped


sntrg = sentence_by_region


def cheatsheet() -> str:
    return "sentence_by_region({}) -> Sentence length by region."
