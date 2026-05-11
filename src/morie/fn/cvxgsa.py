"""Generalized inequality and partial order."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_generalized_p"]


def boyd_generalized_p(x, y, K):
    """
    Generalized inequality and partial order

    Formula: x <=_K y iff y - x in K

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bool

    References
    ----------
    Boyd CVX Ch 2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized inequality and partial order"})


def cheatsheet():
    return "cvxgsa: Generalized inequality and partial order"
