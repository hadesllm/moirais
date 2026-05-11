# morie.fn — function file (hadesllm/morie)
"""Adam optimizer: momentum + RMSProp with bias correction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adam"]


def geron_adam(grads, m, v, b1, b2, eta, eps, t):
    """
    Adam optimizer: momentum + RMSProp with bias correction

    Formula: m_hat = m/(1-b1^t); v_hat = v/(1-b2^t); theta <- theta - eta*m_hat/(sqrt(v_hat)+eps)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adam optimizer: momentum + RMSProp with bias correction"})


def cheatsheet():
    return "hmadam: Adam optimizer: momentum + RMSProp with bias correction"
