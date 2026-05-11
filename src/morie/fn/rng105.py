"""Frequency response of the ideal integrator (DC term aside).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_integrator_frequency_response"]


def rangayyan_ch3_integrator_frequency_response(omega):
    """
    Frequency response of the ideal integrator (DC term aside).

    Formula: H(omega) = 1 / (j*omega)

    Parameters
    ----------
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.116, p. 144
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response of the ideal integrator (DC term aside)."})


def cheatsheet():
    return "rng105: Frequency response of the ideal integrator (DC term aside)."
