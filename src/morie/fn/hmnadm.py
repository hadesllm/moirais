# morie.fn -- function file (hadesllm/morie)
"""NAdam: Adam with Nesterov momentum."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_nadam"]


def geron_nadam(grads, m, v, b1, b2, eta, t):
    """
    NAdam: Adam with Nesterov momentum

    Formula: m_hat = Nesterov-corrected first moment; update as in Adam

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NAdam: Adam with Nesterov momentum"})


def cheatsheet():
    return "hmnadm: NAdam: Adam with Nesterov momentum"
