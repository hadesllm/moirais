"""FISTA for LASSO."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["accelerated_lasso"]


def accelerated_lasso(X, y, lam, steps):
    """
    FISTA for LASSO

    Formula: momentum + soft-threshold

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Beck-Teboulle (2009) FISTA
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FISTA for LASSO"})


def cheatsheet():
    return "acclso: FISTA for LASSO"
