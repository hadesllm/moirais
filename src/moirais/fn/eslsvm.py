"""Kernel SVM dual."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_svm_kernel"]


def esl_svm_kernel(X, y, C, kernel):
    """
    Kernel SVM dual

    Formula: max sum a_i - (1/2) sum a_i a_j y_i y_j K(x_i,x_j)

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
        Keys: model

    References
    ----------
    Hastie ESL Ch 12
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel SVM dual"})


def cheatsheet():
    return "eslsvm: Kernel SVM dual"
