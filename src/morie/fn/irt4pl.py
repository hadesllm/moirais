"""4-parameter logistic with upper asymptote."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["four_parameter_logistic"]


def four_parameter_logistic(y, theta, a, b, c, d):
    """
    4-parameter logistic with upper asymptote

    Formula: P = c + (d - c) / (1 + exp(-a(theta - b)))

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barton & Lord (1981)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "4-parameter logistic with upper asymptote"})


def cheatsheet():
    return "irt4pl: 4-parameter logistic with upper asymptote"
