# morie.fn -- function file (hadesllm/morie)
"""Quantiles for a numeric column in OTIS correctional data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def otis_quantiles(
    df: pd.DataFrame,
    *,
    col: str = "sentence_days",
    probs: list[float] | None = None,
) -> dict:
    """Compute quantiles for a numeric column.

    Parameters
    ----------
    df : DataFrame
        Data with the target numeric column.
    col : str
        Numeric column.
    probs : list of float, optional
        Probability points. Defaults to [0.25, 0.5, 0.75].

    Returns
    -------
    dict
        Keys: quantiles (dict mapping prob->value), n, column.
    """
    if probs is None:
        probs = [0.25, 0.5, 0.75]

    values = df[col].dropna()
    quantiles = {}
    for p in probs:
        quantiles[p] = float(np.quantile(values, p))

    return {
        "quantiles": quantiles,
        "n": len(values),
        "column": col,
    }


def cheatsheet() -> str:
    return "otis_quantiles({}) -> Quantiles for a numeric column in OTIS correctional data."
