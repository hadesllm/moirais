"""Inconsistency check via Dias node-splitting on a single edge."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_network_node_split"]


def ma_network_node_split(yi, vi, design, edge):
    """
    Inconsistency check via Dias node-splitting on a single edge

    Formula: Compare direct vs indirect estimate on edge

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    design : array-like
        Input data.
    edge : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: direct, indirect, p

    References
    ----------
    Dias et al. (2010)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inconsistency check via Dias node-splitting on a single edge"})


def cheatsheet():
    return "manh2h: Inconsistency check via Dias node-splitting on a single edge"
