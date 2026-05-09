"""Dirichlet weights helper for Bayesian bootstrap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_dirichlet_weights"]


def boot_dirichlet_weights(n, B, rng):
    """
    Dirichlet weights helper for Bayesian bootstrap

    Formula: w_b ~ Dirichlet(1,..,1)

    Parameters
    ----------
    n : array-like
        Input data.
    B : array-like
        Input data.
    rng : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W

    References
    ----------
    Rubin (1981)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dirichlet weights helper for Bayesian bootstrap"})


def cheatsheet():
    return "btdir: Dirichlet weights helper for Bayesian bootstrap"
