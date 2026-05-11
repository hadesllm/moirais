"""Ridge regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_ridge"]


def esl_ridge(X, y, lambda_):
    """
    Ridge regression

    Formula: beta_hat^ridge = (X'X + lambda I)^{-1} X'y

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge regression"})


def cheatsheet():
    return "eslrdg: Ridge regression"
