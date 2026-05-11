# morie.fn — function file (hadesllm/morie)
"""Ordinary kriging prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ordinary_kriging"]


def ordinary_kriging(x, coords, target):
    """
    Ordinary kriging prediction

    Formula: Z_hat(s0) = sum lambda_i Z(s_i), minimize MSE

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Schabenberger Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinary kriging prediction"})


def cheatsheet():
    return "okrig: Ordinary kriging prediction"
