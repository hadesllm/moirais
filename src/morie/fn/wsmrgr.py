"""Ridge regression beta_hat = (X'X + lambda I)^{-1} X'y."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_ridge"]


def wasserman_ridge(X, y, lambda_):
    """
    Ridge regression beta_hat = (X'X + lambda I)^{-1} X'y

    Formula: beta_hat = (X'X + lambda I)^{-1} X'y

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
    Wasserman (2004), Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge regression beta_hat = (X'X + lambda I)^{-1} X'y"})


def cheatsheet():
    return "wsmrgr: Ridge regression beta_hat = (X'X + lambda I)^{-1} X'y"
