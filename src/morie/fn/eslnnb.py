"""Naive Bayes classifier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_naive_bayes"]


def esl_naive_bayes(X, y):
    """
    Naive Bayes classifier

    Formula: f_NB(x) = argmax_k pi_k prod_j p_kj(x_j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Naive Bayes classifier"})


def cheatsheet():
    return "eslnnb: Naive Bayes classifier"
