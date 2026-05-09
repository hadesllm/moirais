"""Cross-correlation vector between input vector x(n) and desired response d(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_cross_correlation_vector"]


def rangayyan_ch3_cross_correlation_vector(x, d, n):
    """
    Cross-correlation vector between input vector x(n) and desired response d(n).

    Formula: Theta = E[x(n) * d(n)]

    Parameters
    ----------
    x : array-like
        Input data.
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.160, p. 174
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Cross-correlation vector between input vector x(n) and desired response d(n)."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Cross-correlation vector between input vector x(n) and desired response d(n)."})


def cheatsheet():
    return "rng142: Cross-correlation vector between input vector x(n) and desired response d(n)."
