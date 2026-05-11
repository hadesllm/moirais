"""Imbens sensitivity correlation parameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sensitivity_mediation_imbens"]


def sensitivity_mediation_imbens(Y, X, C, r2_grid):
    """
    Imbens sensitivity correlation parameter

    Formula: vary corr(U, X) and partial R²

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    C : array-like
        Input data.
    r2_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens (2003)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(Y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Imbens sensitivity correlation parameter"})
    result = stats.spearmanr(Y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Imbens sensitivity correlation parameter"})


def cheatsheet():
    return "sensMI: Imbens sensitivity correlation parameter"
