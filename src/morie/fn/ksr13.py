# morie.fn — function file (hadesllm/morie)
"""Tangent space for semiparametric model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_tangent_space"]


def kosorok_tangent_space(x):
    """
    Tangent space for semiparametric model

    Formula: T = closure{score functions along smooth paths}

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
    Kosorok (2008), Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tangent space for semiparametric model"})


def cheatsheet():
    return "ksr13: Tangent space for semiparametric model"
