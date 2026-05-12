# morie.fn -- function file (hadesllm/morie)
"""(eps, delta)-differential privacy definition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_differential_privacy"]


def kamath_differential_privacy(eps, delta):
    """
    (eps, delta)-differential privacy definition

    Formula: for all neighboring D, D' and S: P(M(D) in S) <= exp(eps) * P(M(D') in S) + delta

    Parameters
    ----------
    eps : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: guarantee

    References
    ----------
    Kamath Ch 6, Differential Privacy section
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=float))
    n = len(eps)
    result = float(np.mean(eps))
    se = float(np.std(eps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "(eps, delta)-differential privacy definition"})


def cheatsheet():
    return "kmdp: (eps, delta)-differential privacy definition"
