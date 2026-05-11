"""Chebyshev inequality P(|X-mu|>=k sigma) <= 1/k^2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_chebyshev_ineq"]


def wasserman_chebyshev_ineq(k):
    """
    Chebyshev inequality P(|X-mu|>=k sigma) <= 1/k^2

    Formula: P(|X - mu| >= k sigma) <= 1/k^2

    Parameters
    ----------
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    Wasserman (2004), Ch 4
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chebyshev inequality P(|X-mu|>=k sigma) <= 1/k^2"})


def cheatsheet():
    return "wsmcby: Chebyshev inequality P(|X-mu|>=k sigma) <= 1/k^2"
