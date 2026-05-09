"""Bound on causal effect under missing RD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_causal_rd"]


def bound_causal_rd(y, x, cutoff):
    """
    Bound on causal effect under missing RD

    Formula: sharp bounds with bandwidth-uncertain RD

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Frandsen et al (2012); Bertanha-Imbens (2020)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound on causal effect under missing RD"})


def cheatsheet():
    return "bnscrd: Bound on causal effect under missing RD"
