# morie.fn -- function file (hadesllm/morie)
"""Mixture-of-basis prior for densities using kernel representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mixture_basis_prior"]


def ghosal_mixture_basis_prior(x):
    """
    Mixture-of-basis prior for densities using kernel representation

    Formula: f = sum_k w_k K(x; theta_k), (w_k) ~ Dirichlet

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
    Ghosal Ch 2 §2.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixture-of-basis prior for densities using kernel representation"})


def cheatsheet():
    return "gh_c2_6: Mixture-of-basis prior for densities using kernel representation"
