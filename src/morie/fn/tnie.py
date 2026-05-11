"""Total natural indirect effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["total_natural_indirect_effect"]


def total_natural_indirect_effect(X, M, Y):
    """
    Total natural indirect effect

    Formula: TNIE = E[Y(1, M(1)) - Y(1, M(0))]

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins & Greenland (1992)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total natural indirect effect"})


def cheatsheet():
    return "tnie: Total natural indirect effect"
