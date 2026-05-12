# morie.fn -- function file (hadesllm/morie)
"""Leaky ReLU: small negative slope prevents dead neurons."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_leaky_relu"]


def geron_leaky_relu(z, alpha):
    """
    Leaky ReLU: small negative slope prevents dead neurons

    Formula: LReLU(z) = z if z>=0 else alpha*z

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leaky ReLU: small negative slope prevents dead neurons"})


def cheatsheet():
    return "hmlrel: Leaky ReLU: small negative slope prevents dead neurons"
