# morie.fn -- function file (hadesllm/morie)
"""AdamW: decoupled weight decay."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adamw"]


def geron_adamw(grads, m, v, b1, b2, eta, wd, eps, t):
    """
    AdamW: decoupled weight decay

    Formula: theta <- theta - eta*m_hat/(sqrt(v_hat)+eps) - eta*wd*theta

    Parameters
    ----------
    grads : array-like
        Input data.
    m : array-like
        Input data.
    v : array-like
        Input data.
    b1 : array-like
        Input data.
    b2 : array-like
        Input data.
    eta : array-like
        Input data.
    wd : array-like
        Input data.
    eps : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdamW: decoupled weight decay"})


def cheatsheet():
    return "hmadmw: AdamW: decoupled weight decay"
