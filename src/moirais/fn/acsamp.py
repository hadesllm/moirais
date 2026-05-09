"""Sample autocorrelation function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sample_autocorrelation"]


def sample_autocorrelation(y, max_lag):
    """
    Sample autocorrelation function

    Formula: r_k = sum (y_t - ybar)(y_{t-k} - ybar) / sum (y_t - ybar)^2

    Parameters
    ----------
    y : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Box-Jenkins (1976)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Sample autocorrelation function"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Sample autocorrelation function"})


def cheatsheet():
    return "acsamp: Sample autocorrelation function"
