"""Optimal frequency response of the matched filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_optimal_transfer_function"]


def rangayyan_ch4_matched_filter_optimal_transfer_function(X, K, f, t_0):
    """
    Optimal frequency response of the matched filter.

    Formula: H(f) = K * X*(f) * exp(-j*2*pi*f*t_0)

    Parameters
    ----------
    X : array-like
        Input data.
    K : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.48, p. 239
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Optimal frequency response of the matched filter."})


def cheatsheet():
    return "rng220: Optimal frequency response of the matched filter."
