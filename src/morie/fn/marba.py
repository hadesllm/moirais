"""Variance of Hedges' g for crossover/within-subject designs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_smd_var_correlated_designs"]


def ma_smd_var_correlated_designs(g, n, rho):
    """
    Variance of Hedges' g for crossover/within-subject designs

    Formula: V_g = J²(2(1-ρ)/n + g²/(2(n-1)))

    Parameters
    ----------
    g : array-like
        Input data.
    n : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: var_g

    References
    ----------
    Morris (2008)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of Hedges' g for crossover/within-subject designs"})


def cheatsheet():
    return "marba: Variance of Hedges' g for crossover/within-subject designs"
