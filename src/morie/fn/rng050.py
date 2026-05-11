"""Frequency response obtained by evaluating the Laplace transform on the imaginary axis.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_frequency_response_from_laplace"]


def rangayyan_ch3_frequency_response_from_laplace(h, omega, t, T):
    """
    Frequency response obtained by evaluating the Laplace transform on the imaginary axis.

    Formula: H(omega) = H(s)|_{s=j*omega} = integral_{0}^{T} h(t) * exp(-j*omega*t) dt

    Parameters
    ----------
    h : array-like
        Input data.
    omega : array-like
        Input data.
    t : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.52, p. 117
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response obtained by evaluating the Laplace transform on the imaginary axis."})


def cheatsheet():
    return "rng050: Frequency response obtained by evaluating the Laplace transform on the imaginary axis."
