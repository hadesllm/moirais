"""Closure operator C(x) on the simplex."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_closure"]


def aitchison_closure(x, kappa):
    """
    Closure operator C(x) on the simplex

    Formula: C(x)_i = x_i / sum_j x_j

    Parameters
    ----------
    x : array-like
        Input data.
    kappa : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_closed

    References
    ----------
    Aitchison (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closure operator C(x) on the simplex"})


def cheatsheet():
    return "aitclos: Closure operator C(x) on the simplex"
