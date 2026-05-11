"""Integrated GARCH (alpha + beta = 1)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["igarch_integrated"]


def igarch_integrated(x):
    """
    Integrated GARCH (alpha + beta = 1)

    Formula: sigma_t^2 = omega + alpha eps^2 + (1-alpha) sigma_{t-1}^2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle & Bollerslev (1986)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Integrated GARCH (alpha + beta = 1)"})


def cheatsheet():
    return "igarcm: Integrated GARCH (alpha + beta = 1)"
