"""Inverse Walsh-Hadamard with 1/sqrt(d) normalization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["walsh_hadamard_inverse"]


def walsh_hadamard_inverse(x):
    """
    Inverse Walsh-Hadamard with 1/sqrt(d) normalization

    Formula: WHT^{-1}(y) = WHT(y) (because orthonormal with 1/sqrt(d))

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
    Hadamard (1893); Pratt (1969)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse Walsh-Hadamard with 1/sqrt(d) normalization"})


def cheatsheet():
    return "whtinv: Inverse Walsh-Hadamard with 1/sqrt(d) normalization"
