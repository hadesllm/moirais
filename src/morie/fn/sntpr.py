"""Sentence length percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def sentence_percentiles(
    df: pd.DataFrame,
    *,
    sent_col: str = DEFAULT_COLS["sentence"],
) -> dict:
    """Sentence length at standard percentiles.

    Parameters
    ----------
    df : DataFrame
        Dataset with sentence length column.
    sent_col : str
        Column with sentence length (days).

    Returns
    -------
    dict
        Keys p5, p10, p25, p50, p75, p90, p95, n.
    """
    vals = df[sent_col].dropna().values.astype(float)
    n = len(vals)
    if n == 0:
        return RichResult(payload={f"p{p}": np.nan for p in [5, 10, 25, 50, 75, 90, 95]} | {"n": 0})
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    result = {f"p{p}": float(np.percentile(vals, p)) for p in percentiles}
    result["n"] = n
    return result


sntpr = sentence_percentiles


def cheatsheet() -> str:
    return "sentence_percentiles({}) -> Sentence length percentiles."
