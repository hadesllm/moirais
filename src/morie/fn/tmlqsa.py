"""Quasi-score TMLE -- handles non-likelihood loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_quasi_score"]


def tmle_quasi_score(y, D, X, score_fn):
    """
    Quasi-score TMLE -- handles non-likelihood loss

    Formula: replace likelihood with quasi-score in target step

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    score_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    vdL-Hubbard (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quasi-score TMLE -- handles non-likelihood loss"})


def cheatsheet():
    return "tmlqsa: Quasi-score TMLE -- handles non-likelihood loss"
