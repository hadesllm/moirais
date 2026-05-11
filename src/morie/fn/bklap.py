# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Add-1 (Laplace) smoothing for n-gram probabilities."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_laplace_add_one"]


def burkov_laplace_add_one(counts_ngram, counts_prefix, V):
    """
    Add-1 (Laplace) smoothing for n-gram probabilities

    Formula: P(w_n | w_{1..n-1}) = (count(w_{1..n}) + 1) / (count(w_{1..n-1}) + V)

    Parameters
    ----------
    counts_ngram : array-like
        Input data.
    counts_prefix : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Burkov Ch 2, Laplace (Add-1) Smoothing section
    """
    counts_ngram = np.atleast_1d(np.asarray(counts_ngram, dtype=float))
    n = len(counts_ngram)
    result = float(np.mean(counts_ngram))
    se = float(np.std(counts_ngram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Add-1 (Laplace) smoothing for n-gram probabilities"})


def cheatsheet():
    return "bklap: Add-1 (Laplace) smoothing for n-gram probabilities"
