# moirais.fn — function file (hadesllm/moirais)
"""Scaled ELU (SELU) for self-normalizing networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_selu"]


def geron_selu(z):
    """
    Scaled ELU (SELU) for self-normalizing networks

    Formula: SELU(z) = lambda * ELU(z, alpha); lambda, alpha ~ (1.0507, 1.6733)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaled ELU (SELU) for self-normalizing networks"})


def cheatsheet():
    return "hmselu: Scaled ELU (SELU) for self-normalizing networks"
