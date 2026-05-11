"""Cross-functional correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_correlation"]


def functional_correlation(X, Y):
    """
    Cross-functional correlation

    Formula: normalized inner product of curve pairs

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramsay-Silverman (2005)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(X), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-functional correlation"})
    result = stats.spearmanr(X[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-functional correlation"})


def cheatsheet():
    return "fpccor: Cross-functional correlation"
