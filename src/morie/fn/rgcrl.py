# morie.fn — function file (hadesllm/morie)
"""Correlation dimension (Grassberger-Procaccia)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_correlation_dimension"]


def rangayyan_correlation_dimension(x, y):
    """
    Correlation dimension (Grassberger-Procaccia)

    Formula: C(r) ~ r^D2 as r -> 0

    Parameters
    ----------
    x, y : array-like
        Input series.

    Returns
    -------
    result : dict
        Keys: statistic, p_value, n, method

    References
    ----------
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Correlation dimension (Grassberger-Procaccia)"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Correlation dimension (Grassberger-Procaccia)"})


def cheatsheet():
    return "rgcrl: Correlation dimension (Grassberger-Procaccia)"
