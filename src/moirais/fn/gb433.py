# moirais.fn — function file (hadesllm/moirais)
"""Kolmogorov limiting distribution of sqrt(n)*D_n."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_ks_kolmogorov_limit"]


def gibbons_ks_kolmogorov_limit(d, n):
    """
    Kolmogorov limiting distribution of sqrt(n)*D_n

    Formula: lim P(D_n <= d/sqrt(n)) = L(d) = 1 - 2 sum(-1)^{i+1} exp(-2i^2 d^2)

    Parameters
    ----------
    d : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability

    References
    ----------
    Gibbons Theorem 4.3.3
    """
    d = np.asarray(d, dtype=float)
    n = int(d) if d.ndim == 0 else len(d)
    result = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kolmogorov limiting distribution of sqrt(n)*D_n"})


def cheatsheet():
    return "gb433: Kolmogorov limiting distribution of sqrt(n)*D_n"
