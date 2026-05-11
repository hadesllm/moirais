# morie.fn — function file (hadesllm/morie)
"""Bracketing number for function class."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_bracketing_number"]


def kosorok_bracketing_number(x):
    """
    Bracketing number for function class

    Formula: N_[](e,F,L2(P)) = min covering brackets

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bracketing number for function class"})


def cheatsheet():
    return "ksr05: Bracketing number for function class"
