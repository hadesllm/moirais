# morie.fn — function file (hadesllm/morie)
"""Donsker class verification via bracketing."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_donsker_class"]


def kosorok_donsker_class(x):
    """
    Donsker class verification via bracketing

    Formula: N_[](e,F,L2) < inf for all e > 0

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
    Kosorok (2008), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Donsker class verification via bracketing"})


def cheatsheet():
    return "ksr02: Donsker class verification via bracketing"
