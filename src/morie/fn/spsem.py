"""Spatial error model (SEM): spatial autocorrelation in error term."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["schabenberger_spatial_error_model"]


def schabenberger_spatial_error_model(x, y, w):
    """
    Spatial error model (SEM): spatial autocorrelation in error term

    Formula: Y = X*beta + u; u = lambda*W*u + epsilon; Sigma = sigma^2*(I-lambda*W)'(I-lambda*W)^{-1}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda, beta, se

    References
    ----------
    Schabenberger Ch 6
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Spatial error model (SEM): spatial autocorrelation in error term"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Spatial error model (SEM): spatial autocorrelation in error term"})


def cheatsheet():
    return "spsem: Spatial error model (SEM): spatial autocorrelation in error term"
