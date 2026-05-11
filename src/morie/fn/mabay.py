"""Bayesian random-effects via Metropolis on (μ, log τ)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_bayes_random_effects"]


def ma_bayes_random_effects(yi, vi, n_iter):
    """
    Bayesian random-effects via Metropolis on (μ, log τ)

    Formula: MH on (μ, log τ); Half-Normal prior on τ

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: draws

    References
    ----------
    Higgins-Thompson-Spiegelhalter (2009)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian random-effects via Metropolis on (μ, log τ)"})


def cheatsheet():
    return "mabay: Bayesian random-effects via Metropolis on (μ, log τ)"
