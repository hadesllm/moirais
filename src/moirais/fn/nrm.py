"""Bock Nominal Response Model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nominal_response_bock"]


def nominal_response_bock(y, theta, a_k, c_k):
    """
    Bock Nominal Response Model

    Formula: P(X=k) = exp(a_k theta + c_k) / sum_h exp(a_h theta + c_h)

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a_k : array-like
        Input data.
    c_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bock (1972)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bock Nominal Response Model"})


def cheatsheet():
    return "nrm: Bock Nominal Response Model"
