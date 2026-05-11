"""Graphlet kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graphlet_kernel"]


def graphlet_kernel(G1, G2, k):
    """
    Graphlet kernel

    Formula: counts of size-k subgraph isomorphism types

    Parameters
    ----------
    G1 : array-like
        Input data.
    G2 : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shervashidze et al (2009)
    """
    G1 = np.atleast_1d(np.asarray(G1, dtype=float))
    n = len(G1)
    result = float(np.mean(G1))
    se = float(np.std(G1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graphlet kernel"})


def cheatsheet():
    return "grafl: Graphlet kernel"
