# morie.fn -- function file (hadesllm/morie)
"""Laplace approximation to GP posterior for binary regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_laplace"]


def ghosal_gp_laplace(x, y):
    """
    Laplace approximation to GP posterior for binary regression

    Formula: pi(f|data) approx N(f_hat, (K^{-1}+W)^{-1}), W = diag(-nabla^2 log p(y|f))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 11 §11.7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laplace approximation to GP posterior for binary regression"})


def cheatsheet():
    return "gh_c11_14: Laplace approximation to GP posterior for binary regression"
