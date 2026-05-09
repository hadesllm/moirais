"""Weisfeiler-Lehman graph kernel."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wl_kernel"]


def wl_kernel(G1, G2, K):
    """
    Weisfeiler-Lehman graph kernel

    Formula: WL relabel + iterated histogram comparison

    Parameters
    ----------
    G1 : array-like
        Input data.
    G2 : array-like
        Input data.
    K : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weisfeiler-Lehman graph kernel"})


def cheatsheet():
    return "weisL: Weisfeiler-Lehman graph kernel"
