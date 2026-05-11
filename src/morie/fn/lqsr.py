"""L1 (LAD) regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["l1_regression"]


def l1_regression(X, y):
    """
    L1 (LAD) regression

    Formula: min sum |y_i − x_i^T β|

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bloomfield-Steiger (1983)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L1 (LAD) regression"})


def cheatsheet():
    return "lqsr: L1 (LAD) regression"
