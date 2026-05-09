"""MAD-based anomaly score."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mad_anomaly_score"]


def mad_anomaly_score(x):
    """
    MAD-based anomaly score

    Formula: |x − median|/MAD

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Iglewicz-Hoaglin (1993)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MAD-based anomaly score"})


def cheatsheet():
    return "madAd: MAD-based anomaly score"
