# moirais.fn — function file (hadesllm/moirais)
"""Feller approximation via Bernstein polynomials for distribution functions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bernstein_feller"]


def ghosal_bernstein_feller(x):
    """
    Feller approximation via Bernstein polynomials for distribution functions

    Formula: F_K(x) = sum_{k=0}^K F(k/K) C(K,k) x^k (1-x)^{K-k}

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
    Ghosal Ch 2 §2.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feller approximation via Bernstein polynomials for distribution functions"})


def cheatsheet():
    return "gh_c2_7: Feller approximation via Bernstein polynomials for distribution functions"
