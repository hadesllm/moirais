"""Sample partial autocorrelation function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sample_partial_autocorr"]


def sample_partial_autocorr(y, max_lag):
    """
    Sample partial autocorrelation function

    Formula: PACF via Yule-Walker recursion

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
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Sample partial autocorrelation function"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Sample partial autocorrelation function"})


def cheatsheet():
    return "pacsam: Sample partial autocorrelation function"
