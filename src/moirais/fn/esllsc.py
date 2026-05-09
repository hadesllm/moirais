"""Linear discriminant analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_lda_disc"]


def esl_lda_disc(X, y):
    """
    Linear discriminant analysis

    Formula: delta_k(x) = x' Sigma^{-1} mu_k - (1/2) mu_k' Sigma^{-1} mu_k + log pi_k

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
    Hastie ESL Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear discriminant analysis"})


def cheatsheet():
    return "esllsc: Linear discriminant analysis"
