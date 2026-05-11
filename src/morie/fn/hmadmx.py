# morie.fn — function file (hadesllm/morie)
"""AdaMax: Adam variant using L-infinity norm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adamax"]


def geron_adamax(grads, m, u, b1, b2, eta, t):
    """
    AdaMax: Adam variant using L-infinity norm

    Formula: u <- max(b2*u, |g|); theta <- theta - eta*m_hat/u

    Parameters
    ----------
    grads : array-like
        Input data.
    m : array-like
        Input data.
    u : array-like
        Input data.
    b1 : array-like
        Input data.
    b2 : array-like
        Input data.
    eta : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    result = float(np.mean(grads))
    se = float(np.std(grads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaMax: Adam variant using L-infinity norm"})


def cheatsheet():
    return "hmadmx: AdaMax: Adam variant using L-infinity norm"
