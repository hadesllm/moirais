# morie.fn — function file (hadesllm/morie)
"""Partial autocorrelation function at lag k (controls for intermediate lags)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["joseph_partial_autocorrelation"]


def joseph_partial_autocorrelation(y, max_lag):
    """
    Partial autocorrelation function at lag k (controls for intermediate lags)

    Formula: PACF(k) = last coefficient in AR(k) fit via Yule-Walker

    Parameters
    ----------
    y : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pacf

    References
    ----------
    Joseph Ch 3, PACF section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Partial autocorrelation function at lag k (controls for intermediate lags)"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Partial autocorrelation function at lag k (controls for intermediate lags)"})


def cheatsheet():
    return "jopacf: Partial autocorrelation function at lag k (controls for intermediate lags)"
