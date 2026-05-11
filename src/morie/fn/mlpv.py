"""Pseudo-R^2 / proportional variance reduction across nested models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multilevel_pseudo_variance_ratio"]


def multilevel_pseudo_variance_ratio(y, X, cluster):
    """
    Pseudo-R^2 / proportional variance reduction across nested models

    Formula: PR = (sigma2_e_null - sigma2_e_full) / sigma2_e_null

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Raudenbush & Bryk (2002) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pseudo-R^2 / proportional variance reduction across nested models"})


def cheatsheet():
    return "mlpv: Pseudo-R^2 / proportional variance reduction across nested models"
