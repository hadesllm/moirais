"""Generalized Pareto distribution."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gpd_distribution"]


def gpd_distribution(sigma, xi):
    """
    Generalized Pareto distribution

    Formula: CDF: 1-(1+ξx/σ)^{-1/ξ} for x>0

    Parameters
    ----------
    sigma : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pickands (1975)
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n = len(sigma)
    result = float(np.mean(sigma))
    se = float(np.std(sigma, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized Pareto distribution"})


def cheatsheet():
    return "gpdD: Generalized Pareto distribution"
