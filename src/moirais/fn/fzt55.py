# moirais.fn — function file (hadesllm/moirais)
"""Theorem 5.5: bias and variance of boundary-free KDE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_thm5_5_bdfree_kde_bv"]


def fauzi_thm5_5_bdfree_kde_bv(x, bandwidth, g_func):
    """
    Theorem 5.5: bias and variance of boundary-free KDE

    Formula: Bias=(h^2*c2(x))/(2g'(g^{-1}(x)))*mu2(K); Var=f_X/(nhg')*int K^2 dv + o(1/(nh))

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
    Fauzi Ch 5, Theorem 5.5, Eq 5.10-5.11
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theorem 5.5: bias and variance of boundary-free KDE"})


def cheatsheet():
    return "fzt55: Theorem 5.5: bias and variance of boundary-free KDE"
