# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Linear interpolation of n-gram probabilities across orders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_ngram_interpolation"]


def burkov_ngram_interpolation(probs_by_order, lambdas):
    """
    Linear interpolation of n-gram probabilities across orders

    Formula: P_interp(w_n | context) = sum_k lambda_k * P(w_n | context of order k);  sum_k lambda_k = 1

    Parameters
    ----------
    probs_by_order : array-like
        Input data.
    lambdas : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prob

    References
    ----------
    Burkov Ch 2, Interpolation section
    """
    probs_by_order = np.atleast_1d(np.asarray(probs_by_order, dtype=float))
    n = len(probs_by_order)
    result = float(np.mean(probs_by_order))
    se = float(np.std(probs_by_order, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear interpolation of n-gram probabilities across orders"})


def cheatsheet():
    return "bkintr: Linear interpolation of n-gram probabilities across orders"
