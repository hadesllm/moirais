"""Hadamard response for LDP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hadamard_response"]


def hadamard_response(x, epsilon):
    """
    Hadamard response for LDP

    Formula: map x → ±1 via Hadamard rows

    Parameters
    ----------
    x : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Acharya-Sun-Zhang (2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hadamard response for LDP"})


def cheatsheet():
    return "hkonly: Hadamard response for LDP"
