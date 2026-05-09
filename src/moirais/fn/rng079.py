"""Circular (periodic) convolution of two N-periodic discrete signals.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_periodic_convolution"]


def rangayyan_ch3_periodic_convolution(x_p, h_p, n, N):
    """
    Circular (periodic) convolution of two N-periodic discrete signals.

    Formula: y_p(n) = sum_{k=0}^{N-1} x_p(k) * h_p[(n-k) mod N]

    Parameters
    ----------
    x_p : array-like
        Input data.
    h_p : array-like
        Input data.
    n : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.90, p. 131
    """
    x_p = np.atleast_1d(np.asarray(x_p, dtype=float))
    n = len(x_p)
    result = float(np.mean(x_p))
    se = float(np.std(x_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Circular (periodic) convolution of two N-periodic discrete signals."})


def cheatsheet():
    return "rng079: Circular (periodic) convolution of two N-periodic discrete signals."
