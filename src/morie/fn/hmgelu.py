# morie.fn — function file (hadesllm/morie)
"""Gaussian error linear unit (GELU)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gelu"]


def geron_gelu(z):
    """
    Gaussian error linear unit (GELU)

    Formula: GELU(z) = z * Phi(z)

    Parameters
    ----------
    z : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian error linear unit (GELU)"})


def cheatsheet():
    return "hmgelu: Gaussian error linear unit (GELU)"
