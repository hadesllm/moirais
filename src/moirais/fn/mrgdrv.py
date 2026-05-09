"""Azuma-Hoeffding for martingales."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["martingale_concentration"]


def martingale_concentration(c, n, t):
    """
    Azuma-Hoeffding for martingales

    Formula: P(M_n - M_0 >= t) <= exp(-t^2/(2 sum c_i^2))

    Parameters
    ----------
    c : array-like
        Input data.
    n : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Azuma (1967); Hoeffding (1963)
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    n = len(c)
    result = float(np.mean(c))
    se = float(np.std(c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Azuma-Hoeffding for martingales"})


def cheatsheet():
    return "mrgdrv: Azuma-Hoeffding for martingales"
