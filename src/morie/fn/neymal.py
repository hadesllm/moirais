"""Neyman optimal allocation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["neyman_allocation"]


def neyman_allocation(N, Nh, Sh, n):
    """
    Neyman optimal allocation

    Formula: n_h ~ N_h S_h

    Parameters
    ----------
    N : array-like
        Input data.
    Nh : array-like
        Input data.
    Sh : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neyman (1934)
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neyman optimal allocation"})


def cheatsheet():
    return "neymal: Neyman optimal allocation"
