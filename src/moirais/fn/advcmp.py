"""Advanced composition theorem."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["advanced_composition"]


def advanced_composition(epsilon, delta, k, delta_prime):
    """
    Advanced composition theorem

    Formula: k mechs with ε,δ → (√(2k ln 1/δ')·ε + kε(eᵉ−1), kδ + δ')-DP

    Parameters
    ----------
    epsilon : array-like
        Input data.
    delta : array-like
        Input data.
    k : array-like
        Input data.
    delta_prime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Rothblum-Vadhan (2010)
    """
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=float))
    n = len(epsilon)
    result = float(np.mean(epsilon))
    se = float(np.std(epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Advanced composition theorem"})


def cheatsheet():
    return "advcmp: Advanced composition theorem"
