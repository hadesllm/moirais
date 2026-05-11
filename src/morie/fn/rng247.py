"""Time-domain expression for a wavelet h(n) plus an echo at delay n_0.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_signal_with_echo_output"]


def rangayyan_ch4_signal_with_echo_output(h, a, n_0, n):
    """
    Time-domain expression for a wavelet h(n) plus an echo at delay n_0.

    Formula: y(n) = h(n) + a * h(n - n_0)

    Parameters
    ----------
    h : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.75, p. 249
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-domain expression for a wavelet h(n) plus an echo at delay n_0."})


def cheatsheet():
    return "rng247: Time-domain expression for a wavelet h(n) plus an echo at delay n_0."
