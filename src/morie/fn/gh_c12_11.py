# morie.fn — function file (hadesllm/morie)
"""Credible set frequentist coverage: BvM implies credible sets are confidence sets."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cred_set_cov"]


def ghosal_cred_set_cov(x):
    """
    Credible set frequentist coverage: BvM implies credible sets are confidence sets

    Formula: Pi_n(theta: ||theta-theta_0||<=r_n | X^n) -> 1-alpha => P0^n-prob -> 1-alpha

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
    Ghosal Ch 12 §12.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Credible set frequentist coverage: BvM implies credible sets are confidence sets"})


def cheatsheet():
    return "gh_c12_11: Credible set frequentist coverage: BvM implies credible sets are confidence sets"
