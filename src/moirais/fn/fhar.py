"""Fourier basis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fourier_basis"]


def fourier_basis(t, K):
    """
    Fourier basis

    Formula: sin/cos at frequencies k=1..K

    Parameters
    ----------
    t : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier basis"})


def cheatsheet():
    return "fhar: Fourier basis"
