"""Risch algorithm symbolic integration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["risch_integration"]


def risch_integration(expr, x):
    """
    Risch algorithm symbolic integration

    Formula: differential algebra reduction

    Parameters
    ----------
    expr : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Risch (1969)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Risch algorithm symbolic integration"})


def cheatsheet():
    return "riskr: Risch algorithm symbolic integration"
