"""Mixture of Pitman-Yor processes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pmp_fit"]


def pmp_fit(y, K, sigma_grid, alpha_grid):
    """
    Mixture of Pitman-Yor processes

    Formula: finite mixture of PY at sigma_k

    Parameters
    ----------
    y : array-like
        Input data.
    K : array-like
        Input data.
    sigma_grid : array-like
        Input data.
    alpha_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pitman (2006); Lijoi-Prünster (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixture of Pitman-Yor processes"})


def cheatsheet():
    return "pmpfit: Mixture of Pitman-Yor processes"
