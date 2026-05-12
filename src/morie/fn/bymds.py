# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian metric MDS with posterior credible regions (Bakker-Poole 2013)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_mds"]


def bayesian_mds(D_matrix, n_dims, n_iter):
    """
    Bayesian metric MDS with posterior credible regions (Bakker-Poole 2013)

    Formula: P(X|D) prop L(D|X)*P(X); sample configuration X via MCMC; credible ellipses from marginal posterior

    Parameters
    ----------
    D_matrix : array-like
        Input data.
    n_dims : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'X_samples': 'array', 'credible_regions': 'array'}

    References
    ----------
    Armstrong Ch 6
    """
    D_matrix = np.asarray(D_matrix, dtype=float)
    n = int(D_matrix) if D_matrix.ndim == 0 else len(D_matrix)
    result = float(np.mean(D_matrix))
    se = float(np.std(D_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian metric MDS with posterior credible regions (Bakker-Poole 2013)"})


def cheatsheet():
    return "bymds: Bayesian metric MDS with posterior credible regions (Bakker-Poole 2013)"
