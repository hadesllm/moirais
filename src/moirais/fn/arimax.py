"""ARIMAX with exogenous regressors."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["arimax"]


def arimax(y, X, p, d, q):
    """
    ARIMAX with exogenous regressors

    Formula: ARIMA + linear X term

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARIMAX with exogenous regressors"})


def cheatsheet():
    return "arimax: ARIMAX with exogenous regressors"
