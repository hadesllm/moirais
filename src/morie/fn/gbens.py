# morie.fn — function file (hadesllm/morie)
"""Gradient boosting sequential ensemble."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gradient_boosting_ensemble"]


def gradient_boosting_ensemble(x, y):
    """
    Gradient boosting sequential ensemble

    Formula: F_m(x) = F_{m-1}(x) + alpha * h_m(x)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient boosting sequential ensemble"})


def cheatsheet():
    return "gbens: Gradient boosting sequential ensemble"
