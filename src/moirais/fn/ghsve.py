# moirais.fn — function file (hadesllm/moirais)
"""Sieve prior construction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_sieve_prior"]


def ghosal_sieve_prior(x):
    """
    Sieve prior construction

    Formula: Pi_n = Pi restricted to Theta_n (growing sieves)

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
    Ghosal Ch 9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sieve prior construction"})


def cheatsheet():
    return "ghsve: Sieve prior construction"
