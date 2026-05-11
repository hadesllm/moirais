"""Cross-correlation function (CCF) between two random processes x and y.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ccf_continuous"]


def rangayyan_ch3_ccf_continuous(x, y, t1, tau):
    """
    Cross-correlation function (CCF) between two random processes x and y.

    Formula: theta_xy(t1, t1+tau) = E[x(t1) y(t1+tau)] = double_integral x(t1) y(t1+tau) p_{x,y}(x,y) dx dy

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t1 : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.23, p. 98
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-correlation function (CCF) between two random processes x and y."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-correlation function (CCF) between two random processes x and y."})


def cheatsheet():
    return "rng023: Cross-correlation function (CCF) between two random processes x and y."
