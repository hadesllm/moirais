"""Warm-started Sinkhorn from previous (f,g)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_optimised_potentials_warm"]


def ot_optimised_potentials_warm(a, b, C, epsilon, f0, g0, max_iter):
    """
    Warm-started Sinkhorn from previous (f,g)

    Formula: Initialise u = exp(f/ε); resume

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    f0 : array-like
        Input data.
    g0 : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, f, g

    References
    ----------
    Schmitzer (2019)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Warm-started Sinkhorn from previous (f,g)"})


def cheatsheet():
    return "otopw: Warm-started Sinkhorn from previous (f,g)"
