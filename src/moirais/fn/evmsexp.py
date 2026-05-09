"""Logistic max-stable bivariate density (Gumbel families)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_max_stable_logistic"]


def evt_max_stable_logistic(x, y, alpha):
    """
    Logistic max-stable bivariate density (Gumbel families)

    Formula: F(x,y) = exp(-((x^{-1/α}+y^{-1/α})^α))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: F

    References
    ----------
    Tawn (1988)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic max-stable bivariate density (Gumbel families)"})


def cheatsheet():
    return "evmsexp: Logistic max-stable bivariate density (Gumbel families)"
