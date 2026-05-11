"""Discrete random probability measure on X expressed as an infinite weighted sum of point masses at random locations theta_i with stick-breaking weights W_i.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ghosal_ch3_discrete_random_measure"]


def ghosal_ch3_discrete_random_measure(W_i, theta_i):
    """
    Discrete random probability measure on X expressed as an infinite weighted sum of point masses at random locations theta_i with stick-breaking weights W_i.

    Formula: P = sum_{i=1}^{infty} W_i * delta_{theta_i}

    Parameters
    ----------
    W_i : array-like
        Input data.
    theta_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Ghosal & van der Vaart (2017), Ch 3, Eq 3.10, p. 34
    """
    W_i = np.atleast_1d(np.asarray(W_i, dtype=float))
    n = len(W_i)
    result = float(np.mean(W_i))
    se = float(np.std(W_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete random probability measure on X expressed as an infinite weighted sum of point masses at random locations theta_i with stick-breaking weights W_i."})


def cheatsheet():
    return "ghs017: Discrete random probability measure on X expressed as an infinite weighted sum of point masses at random locations theta_i with stick-breaking weights W_i."
