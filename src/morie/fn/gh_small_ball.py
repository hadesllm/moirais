# morie.fn — function file (hadesllm/morie)
"""Small ball probability for GP: log Pi(||f||<eps) ~ -phi(eps) as eps->0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_small_ball_prob"]


def ghosal_small_ball_prob(x):
    """
    Small ball probability for GP: log Pi(||f||<eps) ~ -phi(eps) as eps->0

    Formula: phi(eps) = log(1/Pi(||f||_infty < eps)) controls prior concentration

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
    Ghosal Ch 11 §11.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Small ball probability for GP: log Pi(||f||<eps) ~ -phi(eps) as eps->0"})


def cheatsheet():
    return "gh_small_ball: Small ball probability for GP: log Pi(||f||<eps) ~ -phi(eps) as eps->0"
