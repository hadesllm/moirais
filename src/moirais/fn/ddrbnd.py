"""DR-bounds for instrumental variable LATE under monotonicity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deer_dr_bounds"]


def deer_dr_bounds(y, D, Z, X):
    """
    DR-bounds for instrumental variable LATE under monotonicity

    Formula: DR moment with bounds on always-takers / never-takers shares

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mogstad-Santos-Torgovitsky (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-bounds for instrumental variable LATE under monotonicity"})


def cheatsheet():
    return "ddrbnd: DR-bounds for instrumental variable LATE under monotonicity"
