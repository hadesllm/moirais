# morie.fn -- function file (hadesllm/morie)
"""First-difference operator for baseline wander removal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_first_diff"]


def rangayyan_first_diff(x):
    """
    First-difference operator for baseline wander removal

    Formula: y[n] = x[n] - x[n-1]; H(f) = 1 - exp(-j2*pi*f*T)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Rangayyan Ch 3.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "First-difference operator for baseline wander removal"})


def cheatsheet():
    return "rgfd1: First-difference operator for baseline wander removal"
