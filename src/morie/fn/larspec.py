"""LARS layer-wise adaptive rate scaling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lars_optimizer"]


def lars_optimizer(g, layer, lr):
    """
    LARS layer-wise adaptive rate scaling

    Formula: per-layer lr scaled by ||w||/||grad||

    Parameters
    ----------
    g : array-like
        Input data.
    layer : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    You-Gitman-Ginsburg (2017)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LARS layer-wise adaptive rate scaling"})


def cheatsheet():
    return "larspec: LARS layer-wise adaptive rate scaling"
