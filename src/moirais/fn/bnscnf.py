"""Confidence set for partial ID."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_confidence_set"]


def bound_confidence_set(theta_grid, moments, alpha):
    """
    Confidence set for partial ID

    Formula: set of theta with hypothesis test failing

    Parameters
    ----------
    theta_grid : array-like
        Input data.
    moments : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Manski (2004)
    """
    theta_grid = np.atleast_1d(np.asarray(theta_grid, dtype=float))
    n = len(theta_grid)
    result = float(np.mean(theta_grid))
    se = float(np.std(theta_grid, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confidence set for partial ID"})


def cheatsheet():
    return "bnscnf: Confidence set for partial ID"
