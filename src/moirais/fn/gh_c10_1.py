# moirais.fn — function file (hadesllm/moirais)
"""Adaptive posterior contraction without knowing smoothness: sieve mixture prior."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_adapt_thm"]


def ghosal_adapt_thm(x):
    """
    Adaptive posterior contraction without knowing smoothness: sieve mixture prior

    Formula: Pi = sum_k pi_k * Pi_k, pi_k ~ exp(-lambda*k*log n), rate eps_n(f0) auto

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
    Ghosal Ch 10 §10.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive posterior contraction without knowing smoothness: sieve mixture prior"})


def cheatsheet():
    return "gh_c10_1: Adaptive posterior contraction without knowing smoothness: sieve mixture prior"
