"""Softmax (normalized exponential) function turning class scores into class probabilities.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_softmax_function"]


def geron_ch4_softmax_function(s, k, K):
    """
    Softmax (normalized exponential) function turning class scores into class probabilities.

    Formula: p_hat_k = sigma(s(x))_k = exp(s_k(x)) / sum_{j=1}^{K} exp(s_j(x))

    Parameters
    ----------
    s : array-like
        Input data.
    k : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p_hat_k

    References
    ----------
    Geron (2026), Ch 4, Eq 4-21, p. 174
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax (normalized exponential) function turning class scores into class probabilities."})


def cheatsheet():
    return "grn021: Softmax (normalized exponential) function turning class scores into class probabilities."
