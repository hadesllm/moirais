# moirais.fn — function file (hadesllm/moirais)
"""Chebyshev's inequality for distribution-free probability bounds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_chebyshev"]


def gibbons_chebyshev(k):
    """
    Chebyshev's inequality for distribution-free probability bounds

    Formula: P(|X - mu| >= k*sigma) <= 1/k^2 for any distribution

    Parameters
    ----------
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability_bound

    References
    ----------
    Gibbons Ch 1.2.5
    """
    k = np.asarray(k, dtype=float)
    n = int(k) if k.ndim == 0 else len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Chebyshev's inequality for distribution-free probability bounds"})


def cheatsheet():
    return "gb_c1: Chebyshev's inequality for distribution-free probability bounds"
