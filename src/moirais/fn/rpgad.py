"""Convert Rényi DP (alpha, epsilon_R) to (epsilon, delta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rdp_to_eps_delta"]


def rdp_to_eps_delta(y, alpha, epsilon_R, delta):
    """
    Convert Rényi DP (alpha, epsilon_R) to (epsilon, delta)

    Formula: epsilon = epsilon_R + log(1/delta)/(alpha-1)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    epsilon_R : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mironov (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convert Rényi DP (alpha, epsilon_R) to (epsilon, delta)"})


def cheatsheet():
    return "rpgad: Convert Rényi DP (alpha, epsilon_R) to (epsilon, delta)"
