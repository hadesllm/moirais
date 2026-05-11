"""Chebyshev polynomial basis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["chebyshev_basis"]


def chebyshev_basis(x, K):
    """
    Chebyshev polynomial basis

    Formula: T_n(x) = cos(n arccos x)

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chebyshev (1853)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chebyshev polynomial basis"})


def cheatsheet():
    return "polyCh: Chebyshev polynomial basis"
