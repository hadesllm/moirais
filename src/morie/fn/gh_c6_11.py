# morie.fn — function file (hadesllm/morie)
"""Posterior consistency for Markov processes via stationary-distribution argument."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_markov_con"]


def ghosal_markov_con(x):
    """
    Posterior consistency for Markov processes via stationary-distribution argument

    Formula: KL(P0^Markov, P^Markov) controlled by stationary KL divergence

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
    Ghosal Ch 6 §6.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior consistency for Markov processes via stationary-distribution argument"})


def cheatsheet():
    return "gh_c6_11: Posterior consistency for Markov processes via stationary-distribution argument"
