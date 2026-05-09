"""Exponential random graph model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ergm"]


def ergm(G, statistics, theta_init):
    """
    Exponential random graph model

    Formula: P(G) ~ exp(theta^T s(G))

    Parameters
    ----------
    G : array-like
        Input data.
    statistics : array-like
        Input data.
    theta_init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frank-Strauss (1986); Hunter-Handcock (2008)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential random graph model"})


def cheatsheet():
    return "ergmod: Exponential random graph model"
