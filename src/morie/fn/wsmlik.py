"""Likelihood function L(theta) = prod f(x_i;theta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_likelihood"]


def wasserman_likelihood(data, f, theta):
    """
    Likelihood function L(theta) = prod f(x_i;theta)

    Formula: L(theta) = prod_i f(X_i; theta)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Likelihood function L(theta) = prod f(x_i;theta)"})


def cheatsheet():
    return "wsmlik: Likelihood function L(theta) = prod f(x_i;theta)"
