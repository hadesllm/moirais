# moirais.fn — function file (hadesllm/moirais)
"""Score standardization (z, T-score, stanine, sten)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_standardize(
    scores: np.ndarray | pd.Series,
    *,
    method: str = "z",
) -> np.ndarray:
    """Standardize scores using the specified transformation.

    Parameters
    ----------
    scores : array-like
        Raw scores.
    method : str
        'z' (mean=0, sd=1), 'T' (mean=50, sd=10),
        'stanine' (mean=5, sd=2, range 1-9),
        'sten' (mean=5.5, sd=2, range 1-10).

    Returns
    -------
    ndarray
        Transformed scores.

    References
    ----------
    Canivez, G. L. (2013). Psychometric versus actuarial interpretation
    of intelligence and related aptitude batteries. In D. H. Saklofske
    et al. (Eds.), The Oxford Handbook of Child Psychological Assessment.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    mu = np.nanmean(s)
    sd = np.nanstd(s, ddof=1)
    if sd < 1e-15:
        sd = 1.0

    z = (s - mu) / sd

    if method == "z":
        return z
    elif method == "T":
        return z * 10.0 + 50.0
    elif method == "stanine":
        t = z * 2.0 + 5.0
        return np.clip(np.round(t), 1, 9)
    elif method == "sten":
        t = z * 2.0 + 5.5
        return np.clip(np.round(t), 1, 10)
    else:
        raise ValueError(f"Unknown method: {method}. Use z/T/stanine/sten.")


def cheatsheet() -> str:
    return "score_standardize({}) -> Score standardization (z, T-score, stanine, sten)."
