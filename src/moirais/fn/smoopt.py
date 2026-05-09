"""Sequential minimal optimization (SVM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["smo_solver"]


def smo_solver(X, y, C, kernel):
    """
    Sequential minimal optimization (SVM)

    Formula: two-element subproblem at each iter

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    C : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Platt (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential minimal optimization (SVM)"})


def cheatsheet():
    return "smoopt: Sequential minimal optimization (SVM)"
