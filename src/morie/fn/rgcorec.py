# morie.fn — function file (hadesllm/morie)
"""Pearson correlation coefficient for morphological analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_correlation_coeff"]


def rangayyan_correlation_coeff(x, y):
    """
    Pearson correlation coefficient for morphological analysis

    Formula: rho = sum((x-mean_x)*(y-mean_y)) / (N*std_x*std_y)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho

    References
    ----------
    Rangayyan Ch 5.4.1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Pearson correlation coefficient for morphological analysis"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Pearson correlation coefficient for morphological analysis"})


def cheatsheet():
    return "rgcorec: Pearson correlation coefficient for morphological analysis"
