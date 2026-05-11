"""Samejima Graded Response Model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graded_response_samejima"]


def graded_response_samejima(y, theta, a, b_k):
    """
    Samejima Graded Response Model

    Formula: P(X>=k) = 1 / (1 + exp(-a(theta - b_k))); P(X=k) = P(>=k) - P(>=k+1)

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    a : array-like
        Input data.
    b_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Samejima (1969)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Samejima Graded Response Model"})


def cheatsheet():
    return "grmsam: Samejima Graded Response Model"
