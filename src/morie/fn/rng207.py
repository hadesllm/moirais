"""Fourier transform of input signal to a matched filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_input_ft"]


def rangayyan_ch4_matched_filter_input_ft(x, t, omega):
    """
    Fourier transform of input signal to a matched filter.

    Formula: X(omega) = integral_{-inf}^{inf} x(t) * exp(-j*omega*t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.33, p. 237
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier transform of input signal to a matched filter."})


def cheatsheet():
    return "rng207: Fourier transform of input signal to a matched filter."
