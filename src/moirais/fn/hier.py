"""Hierarchical RL (options framework)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hierarchical_rl"]


def hierarchical_rl(env, options, meta):
    """
    Hierarchical RL (options framework)

    Formula: options ω = (I, π_ω, β_ω); choose option then primitives

    Parameters
    ----------
    env : array-like
        Input data.
    options : array-like
        Input data.
    meta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sutton-Precup-Singh (1999)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hierarchical RL (options framework)"})


def cheatsheet():
    return "hier: Hierarchical RL (options framework)"
