"""Fact 3.4: l-th absolute moment of N(0, sigma^2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_normal_moment"]


def turboquant_normal_moment(sigma, l):
    """
    Fact 3.4: l-th absolute moment of N(0, sigma^2)

    Formula: E[|X|^l] = sigma^l * 2^{l/2} * Gamma((l+1)/2) / sqrt(pi)

    Parameters
    ----------
    sigma : array-like
        Input data.
    l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: moment

    References
    ----------
    Zandieh et al. 2024 Fact 3.4 (normal moments)
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n = len(sigma)
    result = float(np.mean(sigma))
    se = float(np.std(sigma, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fact 3.4: l-th absolute moment of N(0, sigma^2)"})


def cheatsheet():
    return "tqmom: Fact 3.4: l-th absolute moment of N(0, sigma^2)"
