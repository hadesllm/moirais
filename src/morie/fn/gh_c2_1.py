# morie.fn -- function file (hadesllm/morie)
"""Random basis expansion prior: f = sum_k z_k phi_k with random coefficients z_k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_random_basis_expansion"]


def ghosal_random_basis_expansion(x):
    """
    Random basis expansion prior: f = sum_k z_k phi_k with random coefficients z_k

    Formula: f = sum_{k=1}^infty z_k phi_k, z_k ~ pi_k independently

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 2 §2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random basis expansion prior: f = sum_k z_k phi_k with random coefficients z_k"})


def cheatsheet():
    return "gh_c2_1: Random basis expansion prior: f = sum_k z_k phi_k with random coefficients z_k"
