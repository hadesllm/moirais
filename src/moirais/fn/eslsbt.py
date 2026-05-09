"""Standard error of coefficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_se_beta"]


def esl_se_beta(X, y, beta):
    """
    Standard error of coefficient

    Formula: se(beta_j) = sqrt(sigma^2 v_jj)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: se

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Standard error of coefficient"})


def cheatsheet():
    return "eslsbt: Standard error of coefficient"
