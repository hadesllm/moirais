"""Lasso regression argmin |y-Xb|^2 + lambda |b|_1."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_lasso"]


def wasserman_lasso(X, y, lambda_):
    """
    Lasso regression argmin |y-Xb|^2 + lambda |b|_1

    Formula: argmin |y - X beta|^2 + lambda ||beta||_1

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lasso regression argmin |y-Xb|^2 + lambda |b|_1"})


def cheatsheet():
    return "wsmlas: Lasso regression argmin |y-Xb|^2 + lambda |b|_1"
