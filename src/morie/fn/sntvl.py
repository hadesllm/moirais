"""Sentence length volatility across placements per individual."""

from __future__ import annotations

import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def sentence_volatility(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    id_col: str = DEFAULT_COLS["id"],
) -> pd.DataFrame:
    """Variation in sentence length across placements per individual.

    For each individual with >=2 placements, computes the standard
    deviation, coefficient of variation, and range of sentence lengths.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence and individual ID columns.
    sent_col : str
        Column with sentence length (days).
    id_col : str
        Column with unique individual identifier.

    Returns
    -------
    DataFrame
        Columns: id, mean_sentence, sd_sentence, cv, range_sentence,
        n_placements.
    """
    tmp = df[[id_col, sent_col]].dropna()
    grouped = (
        tmp.groupby(id_col)[sent_col]
        .agg(
            mean_sentence="mean",
            sd_sentence="std",
            min_sentence="min",
            max_sentence="max",
            n_placements="count",
        )
        .reset_index()
    )
    grouped.rename(columns={id_col: "id"}, inplace=True)

    # Only keep individuals with >=2 placements (sd requires >=2)
    grouped = grouped[grouped["n_placements"] >= 2].copy()
    grouped["cv"] = grouped["sd_sentence"] / grouped["mean_sentence"]
    grouped["range_sentence"] = grouped["max_sentence"] - grouped["min_sentence"]
    grouped = grouped.drop(columns=["min_sentence", "max_sentence"])
    return grouped.reset_index(drop=True)


sntvl = sentence_volatility


def cheatsheet() -> str:
    return "sentence_volatility({}) -> Sentence length volatility across placements per individual."
