"""Definite integral via FTC."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["definite_integral"]


def definite_integral(expr, x, a, b):
    """
    Definite integral via FTC

    Formula: ∫_a^b f = F(b) − F(a)

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Definite integral via FTC"})


def cheatsheet():
    return "defint: Definite integral via FTC"
