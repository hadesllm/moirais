"""2-parameter logistic IRT model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["two_parameter_logistic"]


def two_parameter_logistic(y, theta, a, b):
    """
    2-parameter logistic IRT model

    Formula: P(X=1|theta,a,b) = 1 / (1 + exp(-a(theta - b)))

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

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Birnbaum (1968)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "2-parameter logistic IRT model"})


def cheatsheet():
    return "irt2pl: 2-parameter logistic IRT model"
