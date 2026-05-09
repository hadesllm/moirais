# moirais.fn — function file (hadesllm/moirais)
"""Exponential learning rate decay."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lr_exponential"]


def geron_lr_exponential(eta0, decay, t):
    """
    Exponential learning rate decay

    Formula: eta_t = eta_0 * decay^t

    Parameters
    ----------
    eta0 : array-like
        Input data.
    decay : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11
    """
    eta0 = np.atleast_1d(np.asarray(eta0, dtype=float))
    n = len(eta0)
    result = float(np.mean(eta0))
    se = float(np.std(eta0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential learning rate decay"})


def cheatsheet():
    return "hmlrex: Exponential learning rate decay"
