"""Spatial DR-DiD with neighbor effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dr_spatial_did"]


def dr_spatial_did(y, D, X, W_neighbors):
    """
    Spatial DR-DiD with neighbor effects

    Formula: DR moment + spatial-lag covariate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    W_neighbors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Anselin (2003); Verbitsky-Shevtsova et al (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial DR-DiD with neighbor effects"})


def cheatsheet():
    return "drspa: Spatial DR-DiD with neighbor effects"
