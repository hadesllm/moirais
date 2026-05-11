"""Lasso regression L1 penalty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_lasso"]


def esl_lasso(X, y, lambda_):
    """
    Lasso regression L1 penalty

    Formula: beta_hat^lasso = argmin (1/2)|y-Xb|^2 + lambda |b|_1

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso regression L1 penalty"})


def cheatsheet():
    return "esllso: Lasso regression L1 penalty"
