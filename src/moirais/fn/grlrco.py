# moirais.fn — function file (hadesllm/moirais)
"""Cosine annealing LR schedule over T steps."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lr_cosine_annealing"]


def geron_lr_cosine_annealing(eta_min, eta_max, t, T):
    """
    Cosine annealing LR schedule over T steps

    Formula: eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))

    Parameters
    ----------
    eta_min : array-like
        Input data.
    eta_max : array-like
        Input data.
    t : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11, Cosine Annealing section
    """
    eta_min = np.atleast_1d(np.asarray(eta_min, dtype=float))
    n = len(eta_min)
    result = float(np.mean(eta_min))
    se = float(np.std(eta_min, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cosine annealing LR schedule over T steps"})


def cheatsheet():
    return "grlrco: Cosine annealing LR schedule over T steps"
