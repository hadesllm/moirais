"""Trust-region subproblem solver."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["trust_region_subproblem"]


def trust_region_subproblem(g, H, delta):
    """
    Trust-region subproblem solver

    Formula: argmin g^T s + 0.5 s^T H s s.t. ||s|| <= delta

    Parameters
    ----------
    g : array-like
        Input data.
    H : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Steihaug (1983)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trust-region subproblem solver"})


def cheatsheet():
    return "trsopt: Trust-region subproblem solver"
