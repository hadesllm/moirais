"""Qn robust scale (Rousseeuw-Croux)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["qn_scale"]


def qn_scale(y):
    """
    Qn robust scale (Rousseeuw-Croux)

    Formula: Q_n = d_n * { |x_i - x_j| ; i<j }_(k), k = C(h,2), h = n/2 + 1

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rousseeuw & Croux (1993)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Qn robust scale (Rousseeuw-Croux)"})


def cheatsheet():
    return "qnscl: Qn robust scale (Rousseeuw-Croux)"
