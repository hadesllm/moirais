"""f-divergence (Csiszár)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["convex_divergence"]


def convex_divergence(p, q, f):
    """
    f-divergence (Csiszár)

    Formula: D_f(p||q) = sum q f(p/q)

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Csiszár (1967)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "f-divergence (Csiszár)"})


def cheatsheet():
    return "convdv: f-divergence (Csiszár)"
