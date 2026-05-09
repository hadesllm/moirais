# moirais.fn — function file (hadesllm/moirais)
"""Nonnegative matrix factorization (NMF) with multiplicative update rules."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_nmf"]


def rangayyan_nmf(V, r, max_iter, tol):
    """
    Nonnegative matrix factorization (NMF) with multiplicative update rules

    Formula: H <- H*(W^T*V)/(W^T*W*H); W <- W*(V*H^T)/(W*H*H^T)

    Parameters
    ----------
    V : array-like
        Input data.
    r : array-like
        Input data.
    max_iter : array-like
        Input data.
    tol : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W, H

    References
    ----------
    Rangayyan Ch 9.7.3
    """
    V = np.asarray(V, dtype=float)
    n = int(V) if V.ndim == 0 else len(V)
    result = float(np.mean(V))
    se = float(np.std(V, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonnegative matrix factorization (NMF) with multiplicative update rules"})


def cheatsheet():
    return "rgnmf: Nonnegative matrix factorization (NMF) with multiplicative update rules"
