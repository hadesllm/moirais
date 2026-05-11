"""Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_complex_cepstrum_signal_with_echo"]


def rangayyan_ch4_complex_cepstrum_signal_with_echo(h_hat, a, n_0, n):
    """
    Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0).

    Formula: y_hat(n) = h_hat(n) + a*delta(n-n_0) - (a^2/2)*delta(n-2*n_0) + (a^3/3)*delta(n-3*n_0) - ...

    Parameters
    ----------
    h_hat : array-like
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
    Rangayyan (2024), Ch 4, Eq 4.80, p. 249
    """
    h_hat = np.atleast_1d(np.asarray(h_hat, dtype=float))
    n = len(h_hat)
    result = float(np.mean(h_hat))
    se = float(np.std(h_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0)."})


def cheatsheet():
    return "rng252: Complex cepstrum of a signal with a basic wavelet and an echo (impulses at multiples of n_0)."
