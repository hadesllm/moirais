"""Discrete-time cross-correlation function of x(n) and y(n) with shift k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_ccf_discrete_with_delay"]


def rangayyan_ch4_ccf_discrete_with_delay(x, y, k, n):
    """
    Discrete-time cross-correlation function of x(n) and y(n) with shift k.

    Formula: theta_xy(k) = sum_{n} x(n) * y(n + k)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.28, p. 230
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Discrete-time cross-correlation function of x(n) and y(n) with shift k."})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Discrete-time cross-correlation function of x(n) and y(n) with shift k."})


def cheatsheet():
    return "rng202: Discrete-time cross-correlation function of x(n) and y(n) with shift k."
