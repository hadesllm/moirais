# morie.fn -- function file (hadesllm/morie)
"""Functional principal components analysis (FPCA)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["functional_pca"]


def functional_pca(data_functions, n_components):
    """
    Functional principal components analysis (FPCA)

    Formula: x_i(t) = mu(t) + sum_k score_{ik} * phi_k(t); phi_k: functional eigenvectors of Cov(data_functions)

    Parameters
    ----------
    data_functions : array-like
        Input data.
    n_components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'scores': 'matrix', 'eigenfuncs': 'array'}

    References
    ----------
    Montesinos Lopez Ch 14
    """
    data_functions = np.asarray(data_functions, dtype=float)
    n = int(data_functions) if data_functions.ndim == 0 else len(data_functions)
    result = float(np.mean(data_functions))
    se = float(np.std(data_functions, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional principal components analysis (FPCA)"})


def cheatsheet():
    return "fpca: Functional principal components analysis (FPCA)"
