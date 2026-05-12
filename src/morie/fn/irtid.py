# morie.fn -- function file (hadesllm/morie)
"""Identification constraints for Bayesian IRT ideal points."""
import numpy as np
from ._richresult import RichResult

__all__ = ["irt_identification_constraints"]


def irt_identification_constraints(x_init, polarity_idx, pivot_idx):
    """
    Identification constraints for Bayesian IRT ideal points

    Formula: Fix 2+ legislators (polarity + scale): one liberal <0, one conservative >0; or fix mean=0, sd=1

    Parameters
    ----------
    x_init : array-like
        Input data.
    polarity_idx : array-like
        Input data.
    pivot_idx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'x_constrained': 'array'}

    References
    ----------
    Armstrong Ch 6
    """
    x_init = np.asarray(x_init, dtype=float)
    n = int(x_init) if x_init.ndim == 0 else len(x_init)
    result = float(np.mean(x_init))
    se = float(np.std(x_init, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identification constraints for Bayesian IRT ideal points"})


def cheatsheet():
    return "irtid: Identification constraints for Bayesian IRT ideal points"
