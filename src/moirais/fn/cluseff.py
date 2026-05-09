"""Intracluster correlation rho for survey design."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["intracluster_correlation_rho"]


def intracluster_correlation_rho(y, cluster):
    """
    Intracluster correlation rho for survey design

    Formula: DEFF ~ 1 + (n_bar - 1) rho

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kish (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(y), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Intracluster correlation rho for survey design"})
    result = stats.spearmanr(y[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Intracluster correlation rho for survey design"})


def cheatsheet():
    return "cluseff: Intracluster correlation rho for survey design"
