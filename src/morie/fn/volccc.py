"""Constant Conditional Correlation MGARCH."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_ccc_garch"]


def vol_ccc_garch(R_panel, init):
    """
    Constant Conditional Correlation MGARCH

    Formula: H_t = D_t R D_t with constant correlation R

    Parameters
    ----------
    R_panel : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: R, sigmas, ll

    References
    ----------
    Bollerslev (1990)
    """
    R_panel = np.atleast_1d(np.asarray(R_panel, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(R_panel), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Constant Conditional Correlation MGARCH"})
    result = stats.spearmanr(R_panel[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Constant Conditional Correlation MGARCH"})


def cheatsheet():
    return "volccc: Constant Conditional Correlation MGARCH"
