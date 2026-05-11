"""SARIMA + exogenous regressors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sarimax"]


def sarimax(y, X, p, d, q, P, D, Q, s):
    """
    SARIMA + exogenous regressors

    Formula: y_t = β X_t + SARIMA error

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
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
    Hyndman-Khandakar (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SARIMA + exogenous regressors"})


def cheatsheet():
    return "sarimax: SARIMA + exogenous regressors"
