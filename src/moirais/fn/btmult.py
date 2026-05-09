"""Multinomial weights helper for boot resamples."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_multinomial_weights"]


def boot_multinomial_weights(n, B, rng):
    """
    Multinomial weights helper for boot resamples

    Formula: w_b ~ Multinom(n; 1/n,..,1/n)/n

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
    Efron (1979)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multinomial weights helper for boot resamples"})


def cheatsheet():
    return "btmult: Multinomial weights helper for boot resamples"
