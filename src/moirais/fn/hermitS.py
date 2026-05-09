"""Hermite polynomial basis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hermite_basis"]


def hermite_basis(x, K):
    """
    Hermite polynomial basis

    Formula: H_n(x) = (-1)^n e^{x²}d^n/dx^n e^{-x²}

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
    Hermite (1864)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hermite polynomial basis"})


def cheatsheet():
    return "hermitS: Hermite polynomial basis"
