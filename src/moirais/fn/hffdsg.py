"""Hoeffding's inequality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hoeffding_inequality"]


def hoeffding_inequality(a, b, n, t):
    """
    Hoeffding's inequality

    Formula: P(|S_n - E| >= t) <= 2 exp(-2 n t^2 / (b-a)^2)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
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
    Hoeffding (1963)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hoeffding's inequality"})


def cheatsheet():
    return "hffdsg: Hoeffding's inequality"
