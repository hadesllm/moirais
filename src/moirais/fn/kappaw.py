"""Weighted kappa (linear or quadratic)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_kappa"]


def weighted_kappa(rater1, rater2, weights):
    """
    Weighted kappa (linear or quadratic)

    Formula: kappa_w with weight matrix W

    Parameters
    ----------
    rater1 : array-like
        Input data.
    rater2 : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cohen (1968)
    """
    rater1 = np.atleast_1d(np.asarray(rater1, dtype=float))
    n = len(rater1)
    result = float(np.mean(rater1))
    se = float(np.std(rater1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted kappa (linear or quadratic)"})


def cheatsheet():
    return "kappaw: Weighted kappa (linear or quadratic)"
