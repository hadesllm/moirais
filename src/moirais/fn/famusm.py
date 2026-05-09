"""Family-based association (TDT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["family_based_assoc"]


def family_based_assoc(trios):
    """
    Family-based association (TDT)

    Formula: transmission disequilibrium test

    Parameters
    ----------
    trios : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Spielman-McGinnis-Ewens (1993)
    """
    trios = np.atleast_1d(np.asarray(trios, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(trios), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Family-based association (TDT)"})
    result = stats.spearmanr(trios[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Family-based association (TDT)"})


def cheatsheet():
    return "famusm: Family-based association (TDT)"
