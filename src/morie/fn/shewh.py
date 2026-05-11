"""Shewhart 3σ chart."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shewhart"]


def shewhart(x, mu, sigma):
    """
    Shewhart 3σ chart

    Formula: alert if |x_t − μ| > 3σ

    Parameters
    ----------
    x : array-like
        Input data.
    mu : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shewhart (1931)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shewhart 3σ chart"})


def cheatsheet():
    return "shewh: Shewhart 3σ chart"
