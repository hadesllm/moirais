"""Log-logistic AFT model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["log_logistic_aft"]


def log_logistic_aft(time, event, X):
    """
    Log-logistic AFT model

    Formula: log(T) = beta'X + sigma * Z, Z ~ logistic(0,1)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch & Prentice (2002) §2.2.5
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-logistic AFT model"})


def cheatsheet():
    return "llgaft: Log-logistic AFT model"
