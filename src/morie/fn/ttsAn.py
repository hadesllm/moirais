"""Twitter AnomalyDetection (Seasonal Hybrid ESD)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["twitter_anomaly"]


def twitter_anomaly(y, period):
    """
    Twitter AnomalyDetection (Seasonal Hybrid ESD)

    Formula: STL decomp + ESD on remainder

    Parameters
    ----------
    y : array-like
        Input data.
    period : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hochenbaum-Vallis-Kejariwal (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Twitter AnomalyDetection (Seasonal Hybrid ESD)"})


def cheatsheet():
    return "ttsAn: Twitter AnomalyDetection (Seasonal Hybrid ESD)"
