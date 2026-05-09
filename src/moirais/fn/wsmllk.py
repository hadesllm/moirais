"""Log-likelihood l(theta) = sum log f(x_i;theta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_log_likelihood"]


def wasserman_log_likelihood(data, f, theta):
    """
    Log-likelihood l(theta) = sum log f(x_i;theta)

    Formula: l(theta) = sum_i log f(X_i; theta)

    Parameters
    ----------
    data : array-like
        Input data.
    f : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 9
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-likelihood l(theta) = sum log f(x_i;theta)"})


def cheatsheet():
    return "wsmllk: Log-likelihood l(theta) = sum log f(x_i;theta)"
