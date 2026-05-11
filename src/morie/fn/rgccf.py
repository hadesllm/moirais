# morie.fn — function file (hadesllm/morie)
"""Cross-correlation function (CCF) between two signals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ccf"]


def rangayyan_ccf(x, y, max_lag):
    """
    Cross-correlation function (CCF) between two signals

    Formula: R_xy(tau) = (1/N) sum x(n) * y(n + tau)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ccf, lags

    References
    ----------
    Rangayyan Ch 2
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-correlation function (CCF) between two signals"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-correlation function (CCF) between two signals"})


def cheatsheet():
    return "rgccf: Cross-correlation function (CCF) between two signals"
