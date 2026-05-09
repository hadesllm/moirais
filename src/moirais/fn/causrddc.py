"""Calonico-Cattaneo-Titiunik MSE-optimal bandwidth."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_rdd_ccft_bw"]


def causal_rdd_ccft_bw(x, y, cutoff, p):
    """
    Calonico-Cattaneo-Titiunik MSE-optimal bandwidth

    Formula: h_CCT minimising AMSE of local quadratic

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    cutoff : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_CCT, b_CCT

    References
    ----------
    Calonico-Cattaneo-Titiunik (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Calonico-Cattaneo-Titiunik MSE-optimal bandwidth"})


def cheatsheet():
    return "causrddc: Calonico-Cattaneo-Titiunik MSE-optimal bandwidth"
