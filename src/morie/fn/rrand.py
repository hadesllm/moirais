"""Randomized response (LDP for bits)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["randomized_response"]


def randomized_response(bit, epsilon):
    """
    Randomized response (LDP for bits)

    Formula: flip with prob 1/(1+exp(ε))

    Parameters
    ----------
    bit : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Warner (1965)
    """
    bit = np.atleast_1d(np.asarray(bit, dtype=float))
    n = len(bit)
    result = float(np.mean(bit))
    se = float(np.std(bit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Randomized response (LDP for bits)"})


def cheatsheet():
    return "rrand: Randomized response (LDP for bits)"
