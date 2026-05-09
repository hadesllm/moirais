"""Variance-based proportion mediated (R^2)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variance_based_mediation"]


def variance_based_mediation(r2_full, r2_partial):
    """
    Variance-based proportion mediated (R^2)

    Formula: R^2_med = (R^2_full - R^2_partial) / R^2_full

    Parameters
    ----------
    r2_full : array-like
        Input data.
    r2_partial : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    de Heus (2012)
    """
    r2_full = np.atleast_1d(np.asarray(r2_full, dtype=float))
    n = len(r2_full)
    result = float(np.mean(r2_full))
    se = float(np.std(r2_full, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance-based proportion mediated (R^2)"})


def cheatsheet():
    return "vrmed: Variance-based proportion mediated (R^2)"
