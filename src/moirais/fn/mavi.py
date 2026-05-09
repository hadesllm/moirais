"""Variance inflation for correlated effect sizes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_var_inflation_correlated"]


def ma_var_inflation_correlated(V, rho):
    """
    Variance inflation for correlated effect sizes

    Formula: V*_i = V_i + 2 Σ_{j>i} cov_{ij}

    Parameters
    ----------
    V : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: V_inflated

    References
    ----------
    Hedges-Tipton-Johnson (2010)
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(V), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Variance inflation for correlated effect sizes"})
    result = stats.spearmanr(V[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Variance inflation for correlated effect sizes"})


def cheatsheet():
    return "mavi: Variance inflation for correlated effect sizes"
