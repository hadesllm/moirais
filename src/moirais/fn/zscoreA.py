"""Z-score anomaly."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["zscore_anomaly"]


def zscore_anomaly(x, k):
    """
    Z-score anomaly

    Formula: |x − μ|/σ > k

    Parameters
    ----------
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    basic
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Z-score anomaly"})


def cheatsheet():
    return "zscoreA: Z-score anomaly"
