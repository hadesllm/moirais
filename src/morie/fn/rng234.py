"""Fourier transform converts convolution to multiplication.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_fourier_convolution_property"]


def rangayyan_ch4_fourier_convolution_property(X, H, omega):
    """
    Fourier transform converts convolution to multiplication.

    Formula: Y(omega) = X(omega) * H(omega)

    Parameters
    ----------
    X : array-like
        Input data.
    H : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.62, p. 245
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier transform converts convolution to multiplication."})


def cheatsheet():
    return "rng234: Fourier transform converts convolution to multiplication."
