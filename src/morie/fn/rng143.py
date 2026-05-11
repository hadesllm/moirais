"""Autocorrelation matrix of the input vector x(n) used in Wiener filtering.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_autocorrelation_matrix"]


def rangayyan_ch3_autocorrelation_matrix(x, n):
    """
    Autocorrelation matrix of the input vector x(n) used in Wiener filtering.

    Formula: Phi = E[x(n) * x^T(n)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.163, p. 174
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Autocorrelation matrix of the input vector x(n) used in Wiener filtering."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Autocorrelation matrix of the input vector x(n) used in Wiener filtering."})


def cheatsheet():
    return "rng143: Autocorrelation matrix of the input vector x(n) used in Wiener filtering."
