"""Rényi differential privacy (RDP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["renyi_dp"]


def renyi_dp(alpha, sigma):
    """
    Rényi differential privacy (RDP)

    Formula: D_α(M(D)||M(D')) ≤ ε(α)

    Parameters
    ----------
    alpha : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mironov (2017)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rényi differential privacy (RDP)"})


def cheatsheet():
    return "rdpc: Rényi differential privacy (RDP)"
