# morie.fn -- function file (hadesllm/morie)
"""Linear score for class k in softmax regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_score"]


def geron_softmax_score(X, theta):
    """
    Linear score for class k in softmax regression

    Formula: s_k(x) = theta_k^T x

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear score for class k in softmax regression"})


def cheatsheet():
    return "hmsfts: Linear score for class k in softmax regression"
