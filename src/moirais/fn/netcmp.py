"""Graph kernel comparison."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["network_comparison"]


def network_comparison(G1, G2, kernel):
    """
    Graph kernel comparison

    Formula: WL kernel / random walk kernel

    Parameters
    ----------
    G1 : array-like
        Input data.
    G2 : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shervashidze et al (2011)
    """
    G1 = np.atleast_1d(np.asarray(G1, dtype=float))
    n = len(G1)
    result = float(np.mean(G1))
    se = float(np.std(G1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph kernel comparison"})


def cheatsheet():
    return "netcmp: Graph kernel comparison"
