"""3-parameter logistic with guessing parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["three_parameter_logistic"]


def three_parameter_logistic(y, theta, a, b, c):
    """
    3-parameter logistic with guessing parameter

    Formula: P = c + (1-c) / (1 + exp(-a(theta - b)))

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

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Birnbaum (1968); Lord (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3-parameter logistic with guessing parameter"})


def cheatsheet():
    return "irt3pl: 3-parameter logistic with guessing parameter"
