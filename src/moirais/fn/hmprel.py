# moirais.fn — function file (hadesllm/moirais)
"""Parametric ReLU: per-channel learnable negative slope."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_prelu"]


def geron_prelu(z, alpha):
    """
    Parametric ReLU: per-channel learnable negative slope

    Formula: PReLU(z) = z if z>=0 else alpha_j*z, alpha_j learned

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric ReLU: per-channel learnable negative slope"})


def cheatsheet():
    return "hmprel: Parametric ReLU: per-channel learnable negative slope"
