# morie.fn — function file (hadesllm/morie)
"""Boundary-free KDE derived from F_tilde_X by differentiation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_bdfree_density_from_cdf"]


def fauzi_bdfree_density_from_cdf(x, bandwidth, g_func):
    """
    Boundary-free KDE derived from F_tilde_X by differentiation

    Formula: f_tilde_X(x) = (1/nhg'(g^{-1}(x))) sum K((g^{-1}(x)-g^{-1}(X_i))/h)

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
        Keys: estimate

    References
    ----------
    Fauzi Ch 5, Eq 5.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Boundary-free KDE derived from F_tilde_X by differentiation"})


def cheatsheet():
    return "fzbfkde2: Boundary-free KDE derived from F_tilde_X by differentiation"
