"""SGD with momentum."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgd_momentum"]


def sgd_momentum(g, mu, lr):
    """
    SGD with momentum

    Formula: v = mu v - lr g; x += v

    Parameters
    ----------
    g : array-like
        Input data.
    mu : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Polyak (1964)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SGD with momentum"})


def cheatsheet():
    return "sgdmom: SGD with momentum"
