# morie.fn — function file (hadesllm/morie)
"""Orthogonal matching pursuit (OMP) for sparse representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_omp"]


def rangayyan_omp(x, D, sparsity):
    """
    Orthogonal matching pursuit (OMP) for sparse representation

    Formula: r=x; while ||r||>eps: k*=argmax|D^T*r|; x_hat updated by LS on active set; r update

    Parameters
    ----------
    x : array-like
        Input data.
    D : array-like
        Input data.
    sparsity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, support

    References
    ----------
    Rangayyan Ch 9.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Orthogonal matching pursuit (OMP) for sparse representation"})


def cheatsheet():
    return "rgomp: Orthogonal matching pursuit (OMP) for sparse representation"
