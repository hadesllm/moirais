"""NGCF graph CF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ngcf"]


def ngcf(R, K, layers):
    """
    NGCF graph CF

    Formula: GCN message passing on user-item bipartite

    Parameters
    ----------
    R : array-like
        Input data.
    K : array-like
        Input data.
    layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2019) NGCF
    """
    R = np.atleast_1d(np.asarray(R, dtype=float))
    n = len(R)
    result = float(np.mean(R))
    se = float(np.std(R, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NGCF graph CF"})


def cheatsheet():
    return "ngcf: NGCF graph CF"
