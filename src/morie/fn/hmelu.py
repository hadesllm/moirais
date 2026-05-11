# morie.fn — function file (hadesllm/morie)
"""Exponential linear unit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_elu"]


def geron_elu(z, alpha):
    """
    Exponential linear unit

    Formula: ELU(z) = z if z>=0 else alpha*(exp(z)-1)

    Parameters
    ----------
    z : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a

    References
    ----------
    Géron Ch 11
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential linear unit"})


def cheatsheet():
    return "hmelu: Exponential linear unit"
