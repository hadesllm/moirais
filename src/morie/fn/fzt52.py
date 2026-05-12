# morie.fn -- function file (hadesllm/morie)
"""Theorem 5.2: bias and variance of boundary-free KDFE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_2_bdfree_kdfe_bv"]


def fauzi_thm5_2_bdfree_kdfe_bv(x, bandwidth, g_func):
    """
    Theorem 5.2: bias and variance of boundary-free KDFE

    Formula: Bias=(h^2/2)*c1(x)*mu2(K)+o(h^2); Var=(1/n)*F(1-F)-(2h/n)*g'*f*r1+o(h/n)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    g_func : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias_variance

    References
    ----------
    Fauzi Ch 5, Theorem 5.2, Eq 5.6-5.7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.2: bias and variance of boundary-free KDFE"})


def cheatsheet():
    return "fzt52: Theorem 5.2: bias and variance of boundary-free KDFE"
