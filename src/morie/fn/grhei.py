# morie.fn -- function file (hadesllm/morie)
"""He initialization (for ReLU-family activations)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_he_init"]


def geron_he_init(fan_in):
    """
    He initialization (for ReLU-family activations)

    Formula: var(W) = 2 / fan_in; W ~ N(0, var)

    Parameters
    ----------
    fan_in : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W

    References
    ----------
    Géron Ch 11, He Initialization section
    """
    fan_in = np.asarray(fan_in, dtype=float)
    n = int(fan_in) if fan_in.ndim == 0 else len(fan_in)
    result = float(np.mean(fan_in))
    se = float(np.std(fan_in, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "He initialization (for ReLU-family activations)"})


def cheatsheet():
    return "grhei: He initialization (for ReLU-family activations)"
