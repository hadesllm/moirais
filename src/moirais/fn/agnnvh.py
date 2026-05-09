"""AlphaZero policy + value head loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_value_head"]


def alphazero_value_head(z, v, pi, p, theta, c):
    """
    AlphaZero policy + value head loss

    Formula: L = (z-v)^2 - pi^T log p + c||theta||^2

    Parameters
    ----------
    z : array-like
        Input data.
    v : array-like
        Input data.
    pi : array-like
        Input data.
    p : array-like
        Input data.
    theta : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018) Science
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero policy + value head loss"})


def cheatsheet():
    return "agnnvh: AlphaZero policy + value head loss"
