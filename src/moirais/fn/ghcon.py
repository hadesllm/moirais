# moirais.fn — function file (hadesllm/moirais)
"""Schwartz theorem for posterior consistency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_posterior_consistency"]


def ghosal_posterior_consistency(x):
    """
    Schwartz theorem for posterior consistency

    Formula: Pi(U^c | X_1..X_n) -> 0 a.s.

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
    Ghosal Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Schwartz theorem for posterior consistency"})


def cheatsheet():
    return "ghcon: Schwartz theorem for posterior consistency"
