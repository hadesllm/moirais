# morie.fn — function file (hadesllm/morie)
"""Ornstein-Uhlenbeck kernel: k(s,t) = exp(-|s-t|/l), Markovian covariance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_ornstein_uhlenbeck"]


def ghosal_gp_ornstein_uhlenbeck(x):
    """
    Ornstein-Uhlenbeck kernel: k(s,t) = exp(-|s-t|/l), Markovian covariance

    Formula: k(s,t) = sigma^2 exp(-|s-t|/l), stationary Markov GP

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
    Ghosal Ch 11 §11.4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ornstein-Uhlenbeck kernel: k(s,t) = exp(-|s-t|/l), Markovian covariance"})


def cheatsheet():
    return "gh_gp_orn_uhl: Ornstein-Uhlenbeck kernel: k(s,t) = exp(-|s-t|/l), Markovian covariance"
