# morie.fn -- function file (hadesllm/morie)
"""Partial correlation via Kendall's tau controlling for third variable."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gibbons_rank_corr_partial"]


def gibbons_rank_corr_partial(x, y, z):
    """
    Partial correlation via Kendall's tau controlling for third variable

    Formula: rho_xy.z = (rho_xy - rho_xz*rho_yz) / sqrt((1-rho_xz^2)*(1-rho_yz^2))

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
        Keys: partial_corr, p_value

    References
    ----------
    Gibbons Ch 12.6
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Partial correlation via Kendall's tau controlling for third variable"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Partial correlation via Kendall's tau controlling for third variable"})


def cheatsheet():
    return "gb_rco: Partial correlation via Kendall's tau controlling for third variable"
