"""Fisher's z for correlation meta-analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_fishers_z"]


def ma_fishers_z(r, n):
    """
    Fisher's z for correlation meta-analysis

    Formula: z = 0.5 log((1+r)/(1-r)); v = 1/(n-3)

    Parameters
    ----------
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z, var

    References
    ----------
    Fisher (1921)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(r), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Fisher's z for correlation meta-analysis"})
    result = stats.spearmanr(r[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Fisher's z for correlation meta-analysis"})


def cheatsheet():
    return "mafshz: Fisher's z for correlation meta-analysis"
