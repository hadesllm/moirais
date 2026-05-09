# moirais.fn — function file (hadesllm/moirais)
"""Neutral-to-the-right process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_neutral_right"]


def ghosal_neutral_right(x):
    """
    Neutral-to-the-right process

    Formula: F(t) = 1 - prod_{s<=t} (1-dH(s))

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
    Ghosal Ch 14
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neutral-to-the-right process"})


def cheatsheet():
    return "ghntr: Neutral-to-the-right process"
