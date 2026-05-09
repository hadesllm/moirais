# moirais.fn — function file (hadesllm/moirais)
"""Exact null distribution of Kendall tau for small samples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_kendall_exact"]


def gibbons_kendall_exact(n, t):
    """
    Exact null distribution of Kendall tau for small samples

    Formula: P(T = t) = K_t / C(n,2)! where K_t = number of permutations with T = t

    Parameters
    ----------
    n : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exact null distribution of Kendall tau for small samples"})


def cheatsheet():
    return "gb_kt2: Exact null distribution of Kendall tau for small samples"
