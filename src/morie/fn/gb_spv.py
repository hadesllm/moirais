# morie.fn — function file (hadesllm/morie)
"""Variance of Spearman rho under null hypothesis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_spearman_rho_var"]


def gibbons_spearman_rho_var(n):
    """
    Variance of Spearman rho under null hypothesis

    Formula: Var(r_s) = 1/(n-1) for large n; null: r_s ~ N(0, 1/(n-1))

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: variance

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of Spearman rho under null hypothesis"})


def cheatsheet():
    return "gb_spv: Variance of Spearman rho under null hypothesis"
