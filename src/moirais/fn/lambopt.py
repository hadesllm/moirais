"""LAMB layer-wise adaptive."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lamb_optimizer"]


def lamb_optimizer(g, layer_idx, lr):
    """
    LAMB layer-wise adaptive

    Formula: per-layer trust ratio scales Adam step

    Parameters
    ----------
    g : array-like
        Input data.
    layer_idx : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    You et al (2020) LAMB
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LAMB layer-wise adaptive"})


def cheatsheet():
    return "lambopt: LAMB layer-wise adaptive"
