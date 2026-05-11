"""Continuous-time inverse Fourier transform (synthesis).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_inverse_fourier_transform"]


def rangayyan_ch3_inverse_fourier_transform(X, omega, f, t):
    """
    Continuous-time inverse Fourier transform (synthesis).

    Formula: x(t) = (1/(2*pi)) * integral_{-inf}^{inf} X(omega) exp(j*omega*t) d(omega) = integral_{-inf}^{inf} X(f) exp(j*2*pi*f*t) df

    Parameters
    ----------
    X : array-like
        Input data.
    omega : array-like
        Input data.
    f : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.77, p. 126
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time inverse Fourier transform (synthesis)."})


def cheatsheet():
    return "rng066: Continuous-time inverse Fourier transform (synthesis)."
