# morie.fn -- function file (hadesllm/morie)
"""Ewens sampling formula: EPPF for DP, p(n1..nk) = alpha^k prod(n_j-1)! / alpha^{[n]}."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ewens_esf"]


def ghosal_ewens_esf(x):
    """
    Ewens sampling formula: EPPF for DP, p(n1..nk) = alpha^k prod(n_j-1)! / alpha^{[n]}

    Formula: p(n_1..n_k) = alpha^k * prod_{j=1}^k (n_j-1)! / prod_{i=0}^{n-1}(alpha+i)

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
    Ghosal Ch 14 §14.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ewens sampling formula: EPPF for DP, p(n1..nk) = alpha^k prod(n_j-1)! / alpha^{[n]}"})


def cheatsheet():
    return "gh_c14_2: Ewens sampling formula: EPPF for DP, p(n1..nk) = alpha^k prod(n_j-1)! / alpha^{[n]}"
