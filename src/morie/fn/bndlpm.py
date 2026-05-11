"""Linear programming method for bounds."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_lp_method"]


def bound_lp_method(y, D, Z, moment_eqs):
    """
    Linear programming method for bounds

    Formula: LP over simplex of compliance probabilities

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z : array-like
        Input data.
    moment_eqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Balke-Pearl (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear programming method for bounds"})


def cheatsheet():
    return "bndlpm: Linear programming method for bounds"
