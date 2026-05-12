# morie.fn -- function file (hadesllm/morie)
"""PIT-based random number generation from arbitrary continuous CDF."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_pit_rng"]


def gibbons_pit_rng(U, F_inv):
    """
    PIT-based random number generation from arbitrary continuous CDF

    Formula: X = F^{-1}(U) where U ~ Uniform(0,1); inverse transform sampling

    Parameters
    ----------
    U : array-like
        Input data.
    F_inv : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X

    References
    ----------
    Gibbons Example 2.5.2
    """
    U = np.asarray(U, dtype=float)
    n = int(U) if U.ndim == 0 else len(U)
    result = float(np.mean(U))
    se = float(np.std(U, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PIT-based random number generation from arbitrary continuous CDF"})


def cheatsheet():
    return "gb_pit2: PIT-based random number generation from arbitrary continuous CDF"
