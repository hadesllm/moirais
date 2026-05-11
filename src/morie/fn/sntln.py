"""Sentence length distribution summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS


def sentence_length(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
) -> dict:
    """Sentence length distribution statistics.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence length column.
    sent_col : str
        Column with sentence length (days).

    Returns
    -------
    dict
        mean, median, sd, min, max, q25, q75, n.
    """
    vals = df[sent_col].dropna().values.astype(float)
    n = len(vals)
    if n == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "sd": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "n": 0,
        }
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "sd": float(np.std(vals, ddof=1)) if n > 1 else np.nan,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "n": n,
    }


sntln = sentence_length


def cheatsheet() -> str:
    return "sentence_length({}) -> Sentence length distribution summary."
