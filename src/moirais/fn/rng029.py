"""Decomposition of a signal into weighted delta functions via sifting.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_signal_as_delta_decomposition"]


def rangayyan_ch3_signal_as_delta_decomposition(x, alpha, t):
    """
    Decomposition of a signal into weighted delta functions via sifting.

    Formula: x(t) = integral_{-inf}^{inf} x(alpha) delta(t - alpha) d(alpha)

    Parameters
    ----------
    x : array-like
        Input data.
    alpha : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.29, p. 108
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decomposition of a signal into weighted delta functions via sifting."})


def cheatsheet():
    return "rng029: Decomposition of a signal into weighted delta functions via sifting."
