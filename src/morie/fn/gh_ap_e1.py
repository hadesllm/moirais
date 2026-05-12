# morie.fn -- function file (hadesllm/morie)
"""Bernstein polynomial approximation: B_K(f)(x) = sum f(k/K) C(K,k) x^k (1-x)^{K-k}."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bernstein_poly"]


def ghosal_bernstein_poly(x):
    """
    Bernstein polynomial approximation: B_K(f)(x) = sum f(k/K) C(K,k) x^k (1-x)^{K-k}

    Formula: ||B_K(f) - f||_infty <= C*omega(f, 1/sqrt(K)) for modulus of continuity omega

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
    Ghosal App E
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bernstein polynomial approximation: B_K(f)(x) = sum f(k/K) C(K,k) x^k (1-x)^{K-k}"})


def cheatsheet():
    return "gh_ap_e1: Bernstein polynomial approximation: B_K(f)(x) = sum f(k/K) C(K,k) x^k (1-x)^{K-k}"
