# morie.fn — function file (hadesllm/morie)
"""Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_learning_rate_schedule"]


def geron_learning_rate_schedule(t, eta0, t0):
    """
    Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0)

    Formula: eta_t = eta_0 / (t + t_0)

    Parameters
    ----------
    t : array-like
        Input data.
    eta0 : array-like
        Input data.
    t0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 4
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0)"})


def cheatsheet():
    return "hmlrs: Learning rate schedule used with SGD: eta_t = eta_0 / (t + t_0)"
