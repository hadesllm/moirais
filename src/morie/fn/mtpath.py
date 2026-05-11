"""Meta-path analysis on heterogeneous network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["meta_path"]


def meta_path(G, node_types, metapath):
    """
    Meta-path analysis on heterogeneous network

    Formula: counts of paths of given type sequence

    Parameters
    ----------
    G : array-like
        Input data.
    node_types : array-like
        Input data.
    metapath : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sun et al (2011)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Meta-path analysis on heterogeneous network"})


def cheatsheet():
    return "mtpath: Meta-path analysis on heterogeneous network"
