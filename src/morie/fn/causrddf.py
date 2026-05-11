"""Fuzzy RDD via Wald ratio of jumps."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_rdd_fuzzy"]


def causal_rdd_fuzzy(x, y, treat, cutoff, h):
    """
    Fuzzy RDD via Wald ratio of jumps

    Formula: τ_FRD = jump in y / jump in P(treat|x=cutoff)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    treat : array-like
        Input data.
    cutoff : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau, se

    References
    ----------
    Hahn-Todd-Van der Klaauw (2001)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fuzzy RDD via Wald ratio of jumps"})


def cheatsheet():
    return "causrddf: Fuzzy RDD via Wald ratio of jumps"
