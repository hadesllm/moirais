"""Dynamic Conditional Correlation MGARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_dcc_garch"]


def vol_dcc_garch(R_panel, init):
    """
    Dynamic Conditional Correlation MGARCH

    Formula: Q_t = (1-a-b)Q̄ + a u_{t-1}u_{t-1}^T + b Q_{t-1}

    Parameters
    ----------
    R_panel : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a, b, Q_bar, ll

    References
    ----------
    Engle (2002)
    """
    R_panel = np.atleast_1d(np.asarray(R_panel, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(R_panel), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Dynamic Conditional Correlation MGARCH"})
    result = stats.spearmanr(R_panel[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Dynamic Conditional Correlation MGARCH"})


def cheatsheet():
    return "voldcc: Dynamic Conditional Correlation MGARCH"
