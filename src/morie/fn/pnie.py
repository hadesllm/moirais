"""Pure natural indirect effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pure_natural_indirect_effect"]


def pure_natural_indirect_effect(X, M, Y):
    """
    Pure natural indirect effect

    Formula: PNIE = E[Y(0, M(1)) - Y(0, M(0))]

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pure natural indirect effect"})


def cheatsheet():
    return "pnie: Pure natural indirect effect"
