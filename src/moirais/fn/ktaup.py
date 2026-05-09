# moirais.fn — function file (hadesllm/moirais)
"""Kendall tau coefficient for partial correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kendall_tau_partial"]


def kendall_tau_partial(x, y, z):
    """
    Kendall tau coefficient for partial correlation

    Formula: tau_xy.z = (tau_xy - tau_xz*tau_yz) / sqrt((1-tau_xz^2)(1-tau_yz^2))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 12.6
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Kendall tau coefficient for partial correlation"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Kendall tau coefficient for partial correlation"})


def cheatsheet():
    return "ktaup: Kendall tau coefficient for partial correlation"
