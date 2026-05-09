"""Proportion of sentence served."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._otis_const import DEFAULT_COLS


def sentence_served(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
    time_col: str | None = None,
) -> dict:
    """Proportion of sentence served.

    If ``time_col`` is provided, computes time_served / sentence_length
    for each record. Otherwise uses sentence_days as-is and reports
    distribution statistics of sentence lengths (proxy for served time
    when actual time served is unavailable).

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence length column.
    sent_col : str
        Column with total sentence length (days).
    time_col : str, optional
        Column with actual time served (days). If None, reports sentence
        length distribution only.

    Returns
    -------
    dict
        mean_proportion, median_proportion, sd, n. If no time_col,
        proportions are set to NaN and sentence stats are returned instead.
    """
    if time_col is not None and time_col in df.columns:
        tmp = df[[sent_col, time_col]].dropna()
        sent = tmp[sent_col].values.astype(float)
        served = tmp[time_col].values.astype(float)
        mask = sent > 0
        proportions = np.where(mask, served / sent, np.nan)
        proportions = proportions[~np.isnan(proportions)]
        n = len(proportions)
        return {
            "mean_proportion": float(np.mean(proportions)) if n > 0 else np.nan,
            "median_proportion": float(np.median(proportions)) if n > 0 else np.nan,
            "sd": float(np.std(proportions, ddof=1)) if n > 1 else np.nan,
            "n": n,
        }

    vals = df[sent_col].dropna().values.astype(float)
    n = len(vals)
    return {
        "mean_proportion": np.nan,
        "median_proportion": np.nan,
        "sd": np.nan,
        "n": n,
        "mean_sentence": float(np.mean(vals)) if n > 0 else np.nan,
        "median_sentence": float(np.median(vals)) if n > 0 else np.nan,
    }


sntsr = sentence_served


def cheatsheet() -> str:
    return "sentence_served({}) -> Proportion of sentence served."
