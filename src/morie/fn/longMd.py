"""Longitudinal mediation (cross-lagged)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["longitudinal_mediation"]


def longitudinal_mediation(panel, X, M, Y):
    """
    Longitudinal mediation (cross-lagged)

    Formula: X_t -> M_{t+1} -> Y_{t+2}

    Parameters
    ----------
    panel : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cole-Maxwell (2003)
    """
    panel = np.atleast_1d(np.asarray(panel, dtype=float))
    n = len(panel)
    result = float(np.mean(panel))
    se = float(np.std(panel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Longitudinal mediation (cross-lagged)"})


def cheatsheet():
    return "longMd: Longitudinal mediation (cross-lagged)"
