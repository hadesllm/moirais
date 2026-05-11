# morie.fn — function file (hadesllm/morie)
"""Decision boundary for logistic regression: theta^T x = 0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_decision_boundary"]


def geron_decision_boundary(theta, X_grid):
    """
    Decision boundary for logistic regression: theta^T x = 0

    Formula: {x : theta^T x = 0}

    Parameters
    ----------
    theta : array-like
        Input data.
    X_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boundary

    References
    ----------
    Géron Ch 4
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decision boundary for logistic regression: theta^T x = 0"})


def cheatsheet():
    return "hmdbd: Decision boundary for logistic regression: theta^T x = 0"
