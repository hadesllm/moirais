"""Centre study-level predictors on weighted mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_centered_predictors"]


def ma_centered_predictors(X, weights):
    """
    Centre study-level predictors on weighted mean

    Formula: x*_i = x_i - Σw_i x_i/Σw_i

    Parameters
    ----------
    X : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_centered

    References
    ----------
    Borenstein et al. (2009)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Centre study-level predictors on weighted mean"})


def cheatsheet():
    return "mac3: Centre study-level predictors on weighted mean"
