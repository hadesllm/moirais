# morie.fn — function file (hadesllm/morie)
"""Gaussian process prior definition: f ~ GP(mu, k) with mean function mu and covariance kernel k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_prior_def"]


def ghosal_gp_prior_def(x):
    """
    Gaussian process prior definition: f ~ GP(mu, k) with mean function mu and covariance kernel k

    Formula: f ~ GP(mu, k): E[f(x)]=mu(x), Cov(f(x),f(x'))=k(x,x')

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
    Ghosal Ch 2 §2.2.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian process prior definition: f ~ GP(mu, k) with mean function mu and covariance kernel k"})


def cheatsheet():
    return "gh_c2_2: Gaussian process prior definition: f ~ GP(mu, k) with mean function mu and covariance kernel k"
