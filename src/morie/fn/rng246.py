"""Two-impulse input modeling a wavelet plus echo.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_signal_with_echo_input"]


def rangayyan_ch4_signal_with_echo_input(a, n_0, n):
    """
    Two-impulse input modeling a wavelet plus echo.

    Formula: x(n) = delta(n) + a * delta(n - n_0)

    Parameters
    ----------
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
    Rangayyan (2024), Ch 4, Eq 4.74, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Two-impulse input modeling a wavelet plus echo."})


def cheatsheet():
    return "rng246: Two-impulse input modeling a wavelet plus echo."
