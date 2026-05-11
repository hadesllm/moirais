"""Pearson correlation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_pearson_corr"]


def wasserman_pearson_corr(x, y):
    """
    Pearson correlation

    Formula: rho = Cov(X,Y) / (sigma_X sigma_Y)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: correlation

    References
    ----------
    Wasserman (2004), Ch 4
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pearson correlation"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pearson correlation"})


def cheatsheet():
    return "wsmpsn: Pearson correlation"
