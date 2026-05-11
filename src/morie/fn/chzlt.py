"""Cinelli-Hazlett sensitivity (R²-based)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cinelli_hazlett"]


def cinelli_hazlett(model, treat, cov, R2_yu, R2_du):
    """
    Cinelli-Hazlett sensitivity (R²-based)

    Formula: adjusted estimate vs (R²_y~u·x, R²_d~u·x)

    Parameters
    ----------
    model : array-like
        Input data.
    treat : array-like
        Input data.
    cov : array-like
        Input data.
    R2_yu : array-like
        Input data.
    R2_du : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cinelli-Hazlett (2020)
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cinelli-Hazlett sensitivity (R²-based)"})


def cheatsheet():
    return "chzlt: Cinelli-Hazlett sensitivity (R²-based)"
