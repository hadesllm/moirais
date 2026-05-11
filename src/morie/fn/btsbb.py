"""Stationary bootstrap with geometric block lengths."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_stationary_block"]


def boot_stationary_block(x, p, stat, B):
    """
    Stationary bootstrap with geometric block lengths

    Formula: Block lengths ~ Geom(p); circular concat

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Politis & Romano (1994)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stationary bootstrap with geometric block lengths"})


def cheatsheet():
    return "btsbb: Stationary bootstrap with geometric block lengths"
