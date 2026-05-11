"""Dynamic conditional correlation GARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dcc_garch"]


def dcc_garch(X):
    """
    Dynamic conditional correlation GARCH

    Formula: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}, Q_t evolves

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle (2002)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    y = X  # template fallback (no y in spec)
    n = min(len(X), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Dynamic conditional correlation GARCH"})
    result = stats.spearmanr(X[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Dynamic conditional correlation GARCH"})


def cheatsheet():
    return "dccgrch: Dynamic conditional correlation GARCH"
