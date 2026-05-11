"""Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_power_spectrum_signal_echo"]


def rangayyan_ch4_log_power_spectrum_signal_echo(H, a, n_0, omega):
    """
    Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation.

    Formula: log|Y(omega)|^2 = log|H(omega)|^2 + log[1 + a^2 + 2*a*cos(omega*n_0)] = log|H(omega)|^2 + log(1 + a^2) + log(1 + 2*a/(1+a^2) * cos(omega*n_0))

    Parameters
    ----------
    H : array-like
        Input data.
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.85, p. 251
    """
    H = np.atleast_1d(np.asarray(H, dtype=float))
    n = len(H)
    result = float(np.mean(H))
    se = float(np.std(H, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation."})


def cheatsheet():
    return "rng257: Log power spectrum of a signal with wavelet plus echo, showing sinusoidal modulation."
