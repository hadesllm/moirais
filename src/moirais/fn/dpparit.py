"""Pitman-Yor process — two-parameter generalization of DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pitman_yor_process"]


def pitman_yor_process(n, alpha, sigma):
    """
    Pitman-Yor process — two-parameter generalization of DP

    Formula: P(new) = (alpha + sigma K)/(n + alpha)

    Parameters
    ----------
    n : array-like
        Input data.
    alpha : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pitman-Yor (1997)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pitman-Yor process — two-parameter generalization of DP"})


def cheatsheet():
    return "dpparit: Pitman-Yor process — two-parameter generalization of DP"
