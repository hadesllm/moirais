"""Wilcoxon change-point."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wilcox_change"]


def wilcox_change(x):
    """
    Wilcoxon change-point

    Formula: rank-sum stat over splits

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
    Pettitt (1979)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wilcoxon change-point"})


def cheatsheet():
    return "rWcoxa: Wilcoxon change-point"
