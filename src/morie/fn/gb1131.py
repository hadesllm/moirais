# morie.fn -- function file (hadesllm/morie)
"""Spearman rank correlation coefficient r_s."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_spearman_rho"]


def gibbons_spearman_rho(x, y):
    """
    Spearman rank correlation coefficient r_s

    Formula: r_s = 1 - 6*sum(d_i^2)/(n(n^2-1)) where d_i = rank(X_i) - rank(Y_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho, p_value

    References
    ----------
    Gibbons Ch 11.3
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spearman rank correlation coefficient r_s"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Spearman rank correlation coefficient r_s"})


def cheatsheet():
    return "gb1131: Spearman rank correlation coefficient r_s"
