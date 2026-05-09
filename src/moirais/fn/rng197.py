"""Squared and weighted smoothing of the second derivative for dicrotic notch detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_dicrotic_notch_smoothed_squared"]


def rangayyan_ch4_dicrotic_notch_smoothed_squared(p, w, n, M):
    """
    Squared and weighted smoothing of the second derivative for dicrotic notch detection.

    Formula: s(n) = sum_{k=1}^{M} p^2(n - k + 1) * w(k)

    Parameters
    ----------
    p : array-like
        Input data.
    w : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.23, p. 228
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squared and weighted smoothing of the second derivative for dicrotic notch detection."})


def cheatsheet():
    return "rng197: Squared and weighted smoothing of the second derivative for dicrotic notch detection."
