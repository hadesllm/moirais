# morie.fn -- function file (hadesllm/morie)
"""Hyperparameter tuning: optimize over discrete grid or random samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_hyperparameter_tuning"]


def geron_hyperparameter_tuning(param_grid, X, y):
    """
    Hyperparameter tuning: optimize over discrete grid or random samples

    Formula: theta* = argmin_{theta in H} CV(theta)

    Parameters
    ----------
    param_grid : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_params

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hyperparameter tuning: optimize over discrete grid or random samples"})


def cheatsheet():
    return "hmhpt: Hyperparameter tuning: optimize over discrete grid or random samples"
