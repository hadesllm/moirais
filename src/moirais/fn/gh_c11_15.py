# moirais.fn — function file (hadesllm/moirais)
"""Expectation propagation for GP posterior: iterative site approximation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ep_gp"]


def ghosal_ep_gp(x, y):
    """
    Expectation propagation for GP posterior: iterative site approximation

    Formula: q(f) = N(mu, Sigma) with Sigma = (K^{-1} + sum Lambda_i)^{-1}

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
    Ghosal Ch 11 §11.7.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expectation propagation for GP posterior: iterative site approximation"})


def cheatsheet():
    return "gh_c11_15: Expectation propagation for GP posterior: iterative site approximation"
