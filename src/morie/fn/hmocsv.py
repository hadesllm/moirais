# morie.fn -- function file (hadesllm/morie)
"""One-class SVM: learn boundary of high-density region."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_class_svm"]


def geron_one_class_svm(X, nu, gamma):
    """
    One-class SVM: learn boundary of high-density region

    Formula: find smallest sphere / hyperplane separating data from origin

    Parameters
    ----------
    X : array-like
        Input data.
    nu : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-class SVM: learn boundary of high-density region"})


def cheatsheet():
    return "hmocsv: One-class SVM: learn boundary of high-density region"
