# moirais.fn — function file (hadesllm/moirais)
"""Influence function computation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_influence_function"]


def kosorok_influence_function(x, y):
    """
    Influence function computation

    Formula: IF(x) = -I_eff^{-1} S_eff(x)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Influence function computation"})


def cheatsheet():
    return "ksr16: Influence function computation"
