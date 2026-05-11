"""NUTS max-tree-depth saturation rate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tree_depth_saturation"]


def tree_depth_saturation(chains):
    """
    NUTS max-tree-depth saturation rate

    Formula: frac of iterations hitting max_treedepth

    Parameters
    ----------
    chains : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hoffman & Gelman (2014)
    """
    chains = np.atleast_1d(np.asarray(chains, dtype=float))
    n = len(chains)
    result = float(np.mean(chains))
    se = float(np.std(chains, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NUTS max-tree-depth saturation rate"})


def cheatsheet():
    return "trdpd: NUTS max-tree-depth saturation rate"
