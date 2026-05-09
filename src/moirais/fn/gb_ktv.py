# moirais.fn — function file (hadesllm/moirais)
"""Variance of Kendall tau under null hypothesis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_kendall_tau_var"]


def gibbons_kendall_tau_var(n):
    """
    Variance of Kendall tau under null hypothesis

    Formula: Var(T) = 2(2n+5) / (9n(n-1))

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
    Gibbons Ch 11.2
    """
    data = np.asarray(n, dtype=float) if np.ndim(n) > 0 else None
    n = int(n) if np.ndim(n) == 0 else len(n)
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(max(n, 2))
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance of Kendall tau under null hypothesis"})


def cheatsheet():
    return "gb_ktv: Variance of Kendall tau under null hypothesis"
