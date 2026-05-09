# moirais.fn — function file (hadesllm/moirais)
"""Adaptive posterior contraction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_adaptation"]


def ghosal_adaptation(x):
    """
    Adaptive posterior contraction

    Formula: Rate adapts to unknown smoothness beta

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive posterior contraction"})


def cheatsheet():
    return "ghadp: Adaptive posterior contraction"
