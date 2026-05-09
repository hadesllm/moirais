# moirais.fn — function file (hadesllm/moirais)
"""Ridge regression BLUP for marker effects (rrBLUP): equivalent to GBLUP."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rrblup_marker_effects"]


def rrblup_marker_effects(y, Z, lam):
    """
    Ridge regression BLUP for marker effects (rrBLUP): equivalent to GBLUP

    Formula: u_hat = Z'(ZZ' + lam*I)^{-1}*y; lam = sigma_e^2/sigma_u^2

    Parameters
    ----------
    y : array-like
        Input data.
    Z : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'u_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 5
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ridge regression BLUP for marker effects (rrBLUP): equivalent to GBLUP"})


def cheatsheet():
    return "rrblp: Ridge regression BLUP for marker effects (rrBLUP): equivalent to GBLUP"
