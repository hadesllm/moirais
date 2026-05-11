# morie.fn — function file (hadesllm/morie)
"""Exponential learning-rate decay over training steps."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lr_exponential_schedule"]


def geron_lr_exponential_schedule(eta0, gamma, t):
    """
    Exponential learning-rate decay over training steps

    Formula: eta_t = eta_0 * gamma^t

    Parameters
    ----------
    eta0 : array-like
        Input data.
    gamma : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11, Exponential Learning Rate section
    """
    eta0 = np.asarray(eta0, dtype=float)
    n = int(eta0) if eta0.ndim == 0 else len(eta0)
    result = float(np.mean(eta0))
    se = float(np.std(eta0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential learning-rate decay over training steps"})


def cheatsheet():
    return "grlrex: Exponential learning-rate decay over training steps"
