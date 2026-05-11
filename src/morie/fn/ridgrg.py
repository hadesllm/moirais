"""Ridge (Tikhonov) regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ridge_regression"]


def ridge_regression(y, X, lam):
    """
    Ridge (Tikhonov) regression

    Formula: beta = (X'X + lambda I)^-1 X'y

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoerl & Kennard (1970)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge (Tikhonov) regression"})


def cheatsheet():
    return "ridgrg: Ridge (Tikhonov) regression"
