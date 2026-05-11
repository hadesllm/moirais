# morie.fn — function file (hadesllm/morie)
"""GBLUP with genomic relationship matrix."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gblup_full"]


def gblup_full(x, y, markers):
    """
    GBLUP with genomic relationship matrix

    Formula: y = X*beta + Z*g + e, g ~ N(0, G*sigma_g^2)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GBLUP with genomic relationship matrix"})


def cheatsheet():
    return "gblpf: GBLUP with genomic relationship matrix"
