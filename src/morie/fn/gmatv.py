# morie.fn — function file (hadesllm/morie)
"""Genomic relationship matrix (VanRaden methods 1 and 2)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["grm_vanraden"]


def grm_vanraden(markers):
    """
    Genomic relationship matrix (VanRaden methods 1 and 2)

    Formula: G = ZZ' / (2 sum p_j(1-p_j))

    Parameters
    ----------
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 3
    """
    x = np.asarray(markers, dtype=float)
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic relationship matrix (VanRaden methods 1 and 2)"})


def cheatsheet():
    return "gmatv: Genomic relationship matrix (VanRaden methods 1 and 2)"
