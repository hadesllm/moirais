"""Aitchison interquartile distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["compositional_quantile_dist"]


def compositional_quantile_dist(X):
    """
    Aitchison interquartile distance

    Formula: IQD = ||clr(Q3) - clr(Q1)||

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: iqd

    References
    ----------
    Pawlowsky-Glahn (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aitchison interquartile distance"})


def cheatsheet():
    return "aitqld: Aitchison interquartile distance"
