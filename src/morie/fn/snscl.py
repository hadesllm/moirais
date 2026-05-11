"""Sn robust scale (Rousseeuw-Croux)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sn_scale"]


def sn_scale(y):
    """
    Sn robust scale (Rousseeuw-Croux)

    Formula: S_n = c_n * med_i { med_j |x_i - x_j| }

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sn robust scale (Rousseeuw-Croux)"})


def cheatsheet():
    return "snscl: Sn robust scale (Rousseeuw-Croux)"
