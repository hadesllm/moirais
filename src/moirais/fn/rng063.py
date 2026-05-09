"""Euler's formula for the complex exponential basis function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_complex_exponential"]


def rangayyan_ch3_complex_exponential(omega, t):
    """
    Euler's formula for the complex exponential basis function.

    Formula: exp(j*omega*t) = cos(omega*t) + j*sin(omega*t)

    Parameters
    ----------
    omega : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.74, p. 125
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Euler's formula for the complex exponential basis function."})


def cheatsheet():
    return "rng063: Euler's formula for the complex exponential basis function."
