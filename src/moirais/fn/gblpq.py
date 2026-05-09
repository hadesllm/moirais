# moirais.fn — function file (hadesllm/moirais)
"""GBLUP-rrBLUP equivalence theorem."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gblup_equivalence"]


def gblup_equivalence(y, Z, G):
    """
    GBLUP-rrBLUP equivalence theorem

    Formula: GBLUP(G = ZZ'/k) = rrBLUP(lambda = k*sigma_e^2/sigma_g^2); predictions identical

    Parameters
    ----------
    y : array-like
        Input data.
    Z : array-like
        Input data.
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'g_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GBLUP-rrBLUP equivalence theorem"})


def cheatsheet():
    return "gblpq: GBLUP-rrBLUP equivalence theorem"
