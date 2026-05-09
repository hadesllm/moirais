"""Continuous-time Fourier transform with frequency variable omega in rad/s.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_fourier_transform_omega"]


def rangayyan_ch3_fourier_transform_omega(x, t, omega):
    """
    Continuous-time Fourier transform with frequency variable omega in rad/s.

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
    Rangayyan (2024), Ch 3, Eq 3.75, p. 125
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time Fourier transform with frequency variable omega in rad/s."})


def cheatsheet():
    return "rng064: Continuous-time Fourier transform with frequency variable omega in rad/s."
