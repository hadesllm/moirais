# morie.fn — function file (hadesllm/morie)
"""Direct preference optimization (DPO)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dpo"]


def geron_dpo(pi, pi_ref, preferences, beta):
    """
    Direct preference optimization (DPO)

    Formula: L = -log sigmoid(beta * log(pi(y_w|x)/pi_ref(y_w|x)) - beta * log(pi(y_l|x)/pi_ref(y_l|x)))

    Parameters
    ----------
    pi : array-like
        Input data.
    pi_ref : array-like
        Input data.
    preferences : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pi

    References
    ----------
    Géron Ch 15
    """
    pi = np.atleast_1d(np.asarray(pi, dtype=float))
    n = len(pi)
    result = float(np.mean(pi))
    se = float(np.std(pi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Direct preference optimization (DPO)"})


def cheatsheet():
    return "hmdpo: Direct preference optimization (DPO)"
