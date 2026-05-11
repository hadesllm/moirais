# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Add-k smoothing: generalized Laplace with arbitrary pseudo-count k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_add_k_smoothing"]


def burkov_add_k_smoothing(counts_ngram, counts_prefix, V, k):
    """
    Add-k smoothing: generalized Laplace with arbitrary pseudo-count k

    Formula: P(w_n | w_{1..n-1}) = (count(w_{1..n}) + k) / (count(w_{1..n-1}) + k*V)

    Parameters
    ----------
    counts_ngram : array-like
        Input data.
    counts_prefix : array-like
        Input data.
    V : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Burkov Ch 2, Add-k Smoothing section
    """
    counts_ngram = np.atleast_1d(np.asarray(counts_ngram, dtype=float))
    n = len(counts_ngram)
    result = float(np.mean(counts_ngram))
    se = float(np.std(counts_ngram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Add-k smoothing: generalized Laplace with arbitrary pseudo-count k"})


def cheatsheet():
    return "bkaddk: Add-k smoothing: generalized Laplace with arbitrary pseudo-count k"
