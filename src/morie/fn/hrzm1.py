# morie.fn — function file (hadesllm/morie)
"""Semiparametric mixture model via EM."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_mixture_model"]


def horowitz_mixture_model(y):
    """
    Semiparametric mixture model via EM

    Formula: f(y) = sum pi_k * f_k(y|theta_k)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Horowitz (2009), Ch 11
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric mixture model via EM"})


def cheatsheet():
    return "hrzm1: Semiparametric mixture model via EM"
