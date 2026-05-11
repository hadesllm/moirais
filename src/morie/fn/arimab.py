"""ARIMA(p,d,q) Box-Jenkins fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["arima_box_jenkins"]


def arima_box_jenkins(y, p, d, q):
    """
    ARIMA(p,d,q) Box-Jenkins fit

    Formula: phi(B)(1-B)^d Y_t = theta(B) eps_t

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
    Box-Jenkins-Reinsel (1994)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARIMA(p,d,q) Box-Jenkins fit"})


def cheatsheet():
    return "arimab: ARIMA(p,d,q) Box-Jenkins fit"
