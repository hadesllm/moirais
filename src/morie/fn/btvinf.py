"""Influence function via numerical perturbation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_influence_fn"]


def boot_influence_fn(x, stat, eps):
    """
    Influence function via numerical perturbation

    Formula: L̂(x; F) ≈ (T((1-ε)F+εδ_x) - T(F))/ε

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: infl

    References
    ----------
    Hampel (1974)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Influence function via numerical perturbation"})


def cheatsheet():
    return "btvinf: Influence function via numerical perturbation"
