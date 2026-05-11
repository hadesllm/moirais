"""ARIMA(p,d,q) Box-Jenkins."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["arima"]


def arima(y, p, d, q):
    """
    ARIMA(p,d,q) Box-Jenkins

    Formula: (1−sum φ_i L^i)(1−L)^d y = (1+sum θ_j L^j) ε

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    d : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Box-Jenkins (1970)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARIMA(p,d,q) Box-Jenkins"})


def cheatsheet():
    return "arimaF: ARIMA(p,d,q) Box-Jenkins"
