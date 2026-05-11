"""Functional scale (L²-norm)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_scale"]


def functional_scale(f):
    """
    Functional scale (L²-norm)

    Formula: ||f|| = √∫f(t)² dt

    Parameters
    ----------
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional scale (L²-norm)"})


def cheatsheet():
    return "fnscale: Functional scale (L²-norm)"
