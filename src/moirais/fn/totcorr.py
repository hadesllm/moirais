"""Total correlation (multi-info)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["total_correlation"]


def total_correlation(p):
    """
    Total correlation (multi-info)

    Formula: TC(X1..Xn) = sum H(Xi) - H(X1..Xn)

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
    Watanabe (1960)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(p), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Total correlation (multi-info)"})
    result = stats.spearmanr(p[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Total correlation (multi-info)"})


def cheatsheet():
    return "totcorr: Total correlation (multi-info)"
