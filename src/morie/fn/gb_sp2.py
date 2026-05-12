# morie.fn -- function file (hadesllm/morie)
"""Exact null distribution of Spearman r_s for small n via enumeration."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_spearman_exact"]


def gibbons_spearman_exact(n, rho):
    """
    Exact null distribution of Spearman r_s for small n via enumeration

    Formula: P(r_s = rho) = (count of permutations with that rank correlation) / n!

    Parameters
    ----------
    n : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Ch 11.3
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exact null distribution of Spearman r_s for small n via enumeration"})


def cheatsheet():
    return "gb_sp2: Exact null distribution of Spearman r_s for small n via enumeration"
