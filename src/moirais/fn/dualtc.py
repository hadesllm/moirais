"""Dual total correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dual_total_correlation"]


def dual_total_correlation(p):
    """
    Dual total correlation

    Formula: DTC = H(X1..Xn) - sum H(Xi | X_-i)

    Parameters
    ----------
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
 (1978)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(p), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Dual total correlation"})
    result = stats.spearmanr(p[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Dual total correlation"})


def cheatsheet():
    return "dualtc: Dual total correlation"
