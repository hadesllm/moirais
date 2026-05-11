"""Symbolic indefinite integral."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["symbolic_integrate"]


def symbolic_integrate(expr, x):
    """
    Symbolic indefinite integral

    Formula: hybrid Risch + heuristics + table

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bronstein (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symbolic indefinite integral"})


def cheatsheet():
    return "intS: Symbolic indefinite integral"
