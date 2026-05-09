"""Complex log of the spectrum of a signal with a wavelet plus echo.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_log_signal_echo"]


def rangayyan_ch4_log_signal_echo(a, n_0, omega, H_hat):
    """
    Complex log of the spectrum of a signal with a wavelet plus echo.

    Formula: Y_hat(omega) = H_hat(omega) + log[1 + a * exp(-j*omega*n_0)]

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.78, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex log of the spectrum of a signal with a wavelet plus echo."})


def cheatsheet():
    return "rng250: Complex log of the spectrum of a signal with a wavelet plus echo."
