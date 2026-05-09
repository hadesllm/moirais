"""Frequency response of the first-order difference operator.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_first_difference_frequency_response"]


def rangayyan_ch3_first_difference_frequency_response(omega, T):
    """
    Frequency response of the first-order difference operator.

    Formula: H(omega) = (1/T) * [1 - exp(-j*omega)] = (1/T) * exp(-j*omega/2) * [2*j*sin(omega/2)]

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.125, p. 147
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response of the first-order difference operator."})


def cheatsheet():
    return "rng113: Frequency response of the first-order difference operator."
