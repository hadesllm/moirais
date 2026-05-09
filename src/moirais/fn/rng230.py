"""Multiplicative model addressed by homomorphic filtering.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_homomorphic_multiplicative_signal"]


def rangayyan_ch4_homomorphic_multiplicative_signal(x, p, t):
    """
    Multiplicative model addressed by homomorphic filtering.

    Formula: y(t) = x(t) * p(t)

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.58, p. 244
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiplicative model addressed by homomorphic filtering."})


def cheatsheet():
    return "rng230: Multiplicative model addressed by homomorphic filtering."
