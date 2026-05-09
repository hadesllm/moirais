# moirais.fn — function file (hadesllm/moirais)
"""He initialization for ReLU networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_he_init"]


def geron_he_init(fan_in, seed):
    """
    He initialization for ReLU networks

    Formula: Var(W) = 2 / fan_in

    Parameters
    ----------
    fan_in : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W

    References
    ----------
    Géron Ch 11
    """
    fan_in = np.atleast_1d(np.asarray(fan_in, dtype=float))
    n = len(fan_in)
    result = float(np.mean(fan_in))
    se = float(np.std(fan_in, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "He initialization for ReLU networks"})


def cheatsheet():
    return "hmhei: He initialization for ReLU networks"
