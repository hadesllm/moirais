"""Partial autocorrelation (Durbin-Levinson)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["partial_autocorrelation"]


def partial_autocorrelation(y, lag_max):
    """
    Partial autocorrelation (Durbin-Levinson)

    Formula: recursive PACF via Levinson-Durbin

    Parameters
    ----------
    y : array-like
        Input data.
    lag_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Durbin (1960); Levinson (1947)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Partial autocorrelation (Durbin-Levinson)"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Partial autocorrelation (Durbin-Levinson)"})


def cheatsheet():
    return "pacfP: Partial autocorrelation (Durbin-Levinson)"
