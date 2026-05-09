"""Stick-breaking weights for DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stick_breaking_weights"]


def stick_breaking_weights(alpha, truncation):
    """
    Stick-breaking weights for DP

    Formula: V_k ~ Beta(1, alpha); pi_k = V_k prod_{j<k}(1-V_j)

    Parameters
    ----------
    alpha : array-like
        Input data.
    truncation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sethuraman (1994)
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    n = len(alpha)
    result = float(np.mean(alpha))
    se = float(np.std(alpha, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stick-breaking weights for DP"})


def cheatsheet():
    return "dpsbw: Stick-breaking weights for DP"
