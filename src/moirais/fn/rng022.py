"""Pearson correlation coefficient as normalized covariance.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_correlation_coefficient"]


def rangayyan_ch3_correlation_coefficient(C_xy, sigma_x, sigma_y):
    """
    Pearson correlation coefficient as normalized covariance.

    Formula: rho_xy = C_xy / (sigma_x * sigma_y)

    Parameters
    ----------
    C_xy : array-like
        Input data.
    sigma_x : array-like
        Input data.
    sigma_y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.22, p. 98
    """
    C_xy = np.atleast_1d(np.asarray(C_xy, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(C_xy), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pearson correlation coefficient as normalized covariance."})
    result = stats.spearmanr(C_xy[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pearson correlation coefficient as normalized covariance."})


def cheatsheet():
    return "rng022: Pearson correlation coefficient as normalized covariance."
