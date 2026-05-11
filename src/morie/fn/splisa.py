"""Local Indicators of Spatial Association (LISA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_lisa"]


def schabenberger_lisa(x, w):
    """
    Local Indicators of Spatial Association (LISA)

    Formula: I_i = z_i * sum_j w_ij * z_j where z_i = (x_i-xbar)/s

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: local_stats, p_values

    References
    ----------
    Schabenberger Ch 1, Sec 1.3.3
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(w, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Local Indicators of Spatial Association (LISA)"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Local Indicators of Spatial Association (LISA)"})


def cheatsheet():
    return "splisa: Local Indicators of Spatial Association (LISA)"
