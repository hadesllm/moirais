"""Fourier transform of the integral of x(t) including DC term.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_fourier_of_integral"]


def rangayyan_ch3_fourier_of_integral(X, omega):
    """
    Fourier transform of the integral of x(t) including DC term.

    Formula: Y(omega) = (1/(j*omega)) * X(omega) + pi * X(0) * delta(omega)

    Parameters
    ----------
    X : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.115, p. 144
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier transform of the integral of x(t) including DC term."})


def cheatsheet():
    return "rng104: Fourier transform of the integral of x(t) including DC term."
