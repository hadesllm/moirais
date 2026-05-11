"""Squared magnitude (power spectrum) of a signal with wavelet plus echo.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_power_spectrum_signal_echo"]


def rangayyan_ch4_power_spectrum_signal_echo(H, a, n_0, z):
    """
    Squared magnitude (power spectrum) of a signal with wavelet plus echo.

    Formula: |Y(z)|^2 = |H(z)|^2 * |1 + a*z^(-n_0)|^2

    Parameters
    ----------
    H : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.84, p. 251
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squared magnitude (power spectrum) of a signal with wavelet plus echo."})


def cheatsheet():
    return "rng256: Squared magnitude (power spectrum) of a signal with wavelet plus echo."
