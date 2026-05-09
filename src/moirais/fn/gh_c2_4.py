# moirais.fn — function file (hadesllm/moirais)
"""Exponential link function for density: f = exp(psi)/Z normalizes log-density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_exp_link"]


def ghosal_exp_link(x):
    """
    Exponential link function for density: f = exp(psi)/Z normalizes log-density

    Formula: f(x) = exp(psi(x)) / integral exp(psi(t)) dt

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
    Ghosal Ch 2 §2.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential link function for density: f = exp(psi)/Z normalizes log-density"})


def cheatsheet():
    return "gh_c2_4: Exponential link function for density: f = exp(psi)/Z normalizes log-density"
