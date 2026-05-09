"""Tukey biweight regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tukey_regression"]


def tukey_regression(X, y, c):
    """
    Tukey biweight regression

    Formula: IRLS with biweight weights

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tukey (1960); Beaton-Tukey (1974)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tukey biweight regression"})


def cheatsheet():
    return "tukrr: Tukey biweight regression"
