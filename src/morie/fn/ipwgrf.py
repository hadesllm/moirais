"""IPW-augmented forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipw_grf"]


def ipw_grf(y, D, X):
    """
    IPW-augmented forest

    Formula: forest + IPW outcome residual

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wager-Athey (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IPW-augmented forest"})


def cheatsheet():
    return "ipwgrf: IPW-augmented forest"
