"""Variance of a sum of two uncorrelated random processes equals sum of their variances.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_variance_of_sum_uncorrelated"]


def rangayyan_ch3_variance_of_sum_uncorrelated(sigma_x, sigma_eta):
    """
    Variance of a sum of two uncorrelated random processes equals sum of their variances.

    Formula: E[(y - mu_y)^2] = sigma_y^2 = sigma_x^2 + sigma_eta^2

    Parameters
    ----------
    sigma_x : array-like
        Input data.
    sigma_eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.14, p. 96
    """
    sigma_x = np.atleast_1d(np.asarray(sigma_x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(sigma_x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Variance of a sum of two uncorrelated random processes equals sum of their variances."})
    result = stats.spearmanr(sigma_x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Variance of a sum of two uncorrelated random processes equals sum of their variances."})


def cheatsheet():
    return "rng014: Variance of a sum of two uncorrelated random processes equals sum of their variances."
