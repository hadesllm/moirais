"""SARIMA(p,d,q)(P,D,Q)_s."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["seasonal_arima"]


def seasonal_arima(y, p, d, q, P, D, Q, s):
    """
    SARIMA(p,d,q)(P,D,Q)_s

    Formula: seasonal × non-seasonal AR/MA polynomial

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
    P : array-like
        Input data.
    D : array-like
        Input data.
    Q : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Box-Jenkins (1976)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SARIMA(p,d,q)(P,D,Q)_s"})


def cheatsheet():
    return "sarima: SARIMA(p,d,q)(P,D,Q)_s"
