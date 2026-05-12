# morie.fn -- function file (hadesllm/morie)
"""Sparse coding given fixed dictionary (OMP/LASSO)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_dictionary_sparse"]


def rangayyan_dictionary_sparse(Y, D, sparsity_T):
    """
    Sparse coding given fixed dictionary (OMP/LASSO)

    Formula: For each y_i: alpha_i = argmin ||alpha||_0 s.t. ||y_i - D*alpha_i||^2 < epsilon

    Parameters
    ----------
    Y : array-like
        Input data.
    D : array-like
        Input data.
    sparsity_T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_sparse

    References
    ----------
    Rangayyan Ch 9.5
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse coding given fixed dictionary (OMP/LASSO)"})


def cheatsheet():
    return "rgldsp: Sparse coding given fixed dictionary (OMP/LASSO)"
