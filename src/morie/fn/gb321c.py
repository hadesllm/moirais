# morie.fn — function file (hadesllm/morie)
"""Marginal distribution of R1 (runs of type 1)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_marginal_r1"]


def gibbons_marginal_r1(r1, n1, n2):
    """
    Marginal distribution of R1 (runs of type 1)

    Formula: f_{R1}(r1) = C(n1-1,r1-1)*C(n2+1,r1) / C(n1+n2, n1)

    Parameters
    ----------
    r1 : array-like
        Input data.
    n1 : array-like
        Input data.
    n2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Corollary 3.2.1
    """
    r1 = np.asarray(r1, dtype=float)
    n = int(r1) if r1.ndim == 0 else len(r1)
    result = float(np.mean(r1))
    se = float(np.std(r1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Marginal distribution of R1 (runs of type 1)"})


def cheatsheet():
    return "gb321c: Marginal distribution of R1 (runs of type 1)"
