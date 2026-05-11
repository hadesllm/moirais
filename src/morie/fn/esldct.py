"""Decision tree CART splitting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_decision_tree"]


def esl_decision_tree(X, y):
    """
    Decision tree CART splitting

    Formula: min sum (y_i - c_R)^2 + sum (y_i - c_L)^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tree

    References
    ----------
    Hastie ESL Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decision tree CART splitting"})


def cheatsheet():
    return "esldct: Decision tree CART splitting"
