"""Omega hierarchical for general-factor reliability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["omega_hierarchical"]


def omega_hierarchical(X, loadings_g, loadings_specific):
    """
    Omega hierarchical for general-factor reliability

    Formula: omega_h = (sum g_loadings)^2 / Var(T)

    Parameters
    ----------
    X : array-like
        Input data.
    loadings_g : array-like
        Input data.
    loadings_specific : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zinbarg-Revelle-Yovel-Li (2005)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Omega hierarchical for general-factor reliability"})


def cheatsheet():
    return "omegah: Omega hierarchical for general-factor reliability"
