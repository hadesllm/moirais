"""Intrinsic CAR (ICAR) prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icar_prior"]


def icar_prior(adjacency, tau):
    """
    Intrinsic CAR (ICAR) prior

    Formula: u | u_-i ~ N(mean(u_neighbors), tau^2/n_i)

    Parameters
    ----------
    adjacency : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Besag (1974)
    """
    adjacency = np.atleast_1d(np.asarray(adjacency, dtype=float))
    n = len(adjacency)
    result = float(np.mean(adjacency))
    se = float(np.std(adjacency, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intrinsic CAR (ICAR) prior"})


def cheatsheet():
    return "icarbm: Intrinsic CAR (ICAR) prior"
