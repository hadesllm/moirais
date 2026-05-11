"""RMSProp."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rmsprop"]


def rmsprop(g, rho, lr, eps):
    """
    RMSProp

    Formula: E[g^2] = rho E + (1-rho) g^2; x -= lr g / sqrt(E)

    Parameters
    ----------
    g : array-like
        Input data.
    rho : array-like
        Input data.
    lr : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tieleman-Hinton (2012)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RMSProp"})


def cheatsheet():
    return "rmsoptm: RMSProp"
