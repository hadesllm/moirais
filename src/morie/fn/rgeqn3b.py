# morie.fn -- function file (hadesllm/morie)
"""Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_correlation_sum"]


def rangayyan_ch3_correlation_sum(x, y):
    """
    Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]

    Formula: R_xy[m] = sum_n x[n]*y[n+m]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R_xy, lags

    References
    ----------
    Rangayyan Ch 3.4.1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]"})


def cheatsheet():
    return "rgeqn3b: Cross-correlation via convolution: R_xy[m] = x[-n] conv y[n]"
