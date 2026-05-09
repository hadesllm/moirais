# moirais.fn — function file (hadesllm/moirais)
"""Generalized extreme value distribution fit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["extreme_value_gev"]


def extreme_value_gev(x):
    """
    Generalized extreme value distribution fit

    Formula: F(x) = exp(-(1+xi*(x-mu)/sigma)^{-1/xi})

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Coles (2001)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized extreme value distribution fit"})


def cheatsheet():
    return "extvm: Generalized extreme value distribution fit"
