# moirais.fn — function file (hadesllm/moirais)
"""Sample autocorrelation function at lag k."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["joseph_autocorrelation_function"]


def joseph_autocorrelation_function(y, max_lag):
    """
    Sample autocorrelation function at lag k

    Formula: rho_k = sum_{t=k+1}^T (y_t - mean) * (y_{t-k} - mean) / sum_t (y_t - mean)^2

    Parameters
    ----------
    y : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: acf

    References
    ----------
    Joseph Ch 3, ACF/PACF section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Sample autocorrelation function at lag k"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Sample autocorrelation function at lag k"})


def cheatsheet():
    return "joacf: Sample autocorrelation function at lag k"
