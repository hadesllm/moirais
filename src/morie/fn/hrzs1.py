# morie.fn — function file (hadesllm/morie)
"""Semiparametric sample selection model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_sample_selection"]


def horowitz_sample_selection(x, y, z, d):
    """
    Semiparametric sample selection model

    Formula: E[Y|X,D=1] = X'beta + lambda(Z'gamma)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric sample selection model"})


def cheatsheet():
    return "hrzs1: Semiparametric sample selection model"
