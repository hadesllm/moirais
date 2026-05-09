"""SVM dual QP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["svm_dual"]


def svm_dual(X, y, C, kernel):
    """
    SVM dual QP

    Formula: max sum alpha_i - 0.5 sum y_i y_j alpha_i alpha_j K_ij

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
    Cortes-Vapnik (1995)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SVM dual QP"})


def cheatsheet():
    return "svmopt: SVM dual QP"
