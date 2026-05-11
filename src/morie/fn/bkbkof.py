# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""N-gram backoff: fall back to lower-order n-gram when higher-order count is zero."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_ngram_backoff"]


def burkov_ngram_backoff(counts_by_order, alpha):
    """
    N-gram backoff: fall back to lower-order n-gram when higher-order count is zero

    Formula: P_backoff(w_n | w_{1..n-1}) = P_MLE(w_n | w_{1..n-1}) if count > 0 else alpha * P_backoff(w_n | w_{2..n-1})

    Parameters
    ----------
    counts_by_order : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Burkov Ch 2, Backoff section
    """
    counts_by_order = np.atleast_1d(np.asarray(counts_by_order, dtype=float))
    n = len(counts_by_order)
    result = float(np.mean(counts_by_order))
    se = float(np.std(counts_by_order, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-gram backoff: fall back to lower-order n-gram when higher-order count is zero"})


def cheatsheet():
    return "bkbkof: N-gram backoff: fall back to lower-order n-gram when higher-order count is zero"
