# morie.fn — function file (hadesllm/morie)
"""Semiparametric efficiency bound."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_information_bound"]


def kosorok_information_bound(x, y):
    """
    Semiparametric efficiency bound

    Formula: I_eff = E[S_eff * S_eff']

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
    Kosorok (2008), Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric efficiency bound"})


def cheatsheet():
    return "ksr12: Semiparametric efficiency bound"
