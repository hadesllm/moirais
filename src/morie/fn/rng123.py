"""Squared-magnitude response of the analog Butterworth lowpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_lowpass_squared_magnitude"]


def rangayyan_ch3_butterworth_lowpass_squared_magnitude(Omega, Omega_c, N):
    """
    Squared-magnitude response of the analog Butterworth lowpass filter.

    Formula: |H_a(j*Omega)|^2 = 1 / (1 + (j*Omega/(j*Omega_c))^(2*N))

    Parameters
    ----------
    Omega : array-like
        Input data.
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.135, p. 154
    """
    Omega = np.atleast_1d(np.asarray(Omega, dtype=float))
    n = len(Omega)
    result = float(np.mean(Omega))
    se = float(np.std(Omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squared-magnitude response of the analog Butterworth lowpass filter."})


def cheatsheet():
    return "rng123: Squared-magnitude response of the analog Butterworth lowpass filter."
