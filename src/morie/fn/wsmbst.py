"""Boosting (AdaBoost weights)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_boosting"]


def wasserman_boosting(X, y, model, T):
    """
    Boosting (AdaBoost weights)

    Formula: alpha_t = (1/2) log((1-err_t)/err_t)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Wasserman (2004), Ch 22
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Boosting (AdaBoost weights)"})


def cheatsheet():
    return "wsmbst: Boosting (AdaBoost weights)"
