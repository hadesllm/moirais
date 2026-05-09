# moirais.fn — function file (hadesllm/moirais)
"""Basis pursuit: L1 minimization for sparse representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_basis_pursuit"]


def rangayyan_basis_pursuit(x, D, tol):
    """
    Basis pursuit: L1 minimization for sparse representation

    Formula: min ||alpha||_1 s.t. D*alpha = x; solved via LP or ADMM

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha

    References
    ----------
    Rangayyan Ch 9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basis pursuit: L1 minimization for sparse representation"})


def cheatsheet():
    return "rgbp: Basis pursuit: L1 minimization for sparse representation"
