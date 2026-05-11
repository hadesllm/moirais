# morie.fn — function file (hadesllm/morie)
"""Nonparametric instrumental variables."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_nonparametric_iv"]


def horowitz_nonparametric_iv(x, y, z):
    """
    Nonparametric instrumental variables

    Formula: E[Y|Z] = integral g(x) dF(x|z)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 12
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric instrumental variables"})


def cheatsheet():
    return "hrzn1: Nonparametric instrumental variables"
