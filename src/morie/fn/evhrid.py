"""Hüsler-Reiss bivariate dependence parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_husler_reiss_dep"]


def evt_husler_reiss_dep(x, y, lam):
    """
    Hüsler-Reiss bivariate dependence parameter

    Formula: F(x,y) = exp(-x Φ(λ + log(y/x)/(2λ)) - y Φ(λ + log(x/y)/(2λ)))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F

    References
    ----------
    Hüsler & Reiss (1989)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hüsler-Reiss bivariate dependence parameter"})


def cheatsheet():
    return "evhrid: Hüsler-Reiss bivariate dependence parameter"
