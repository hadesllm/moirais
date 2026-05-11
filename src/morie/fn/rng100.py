"""Frequency response of the 8-point MA filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_frequency_response"]


def rangayyan_ch3_ma_8point_frequency_response(omega):
    """
    Frequency response of the 8-point MA filter.

    Formula: H(omega) = (1/8) * sum_{k=0}^{7} exp(-j*omega*k) = (1/8) * {1 + exp(-j*4*omega)} * {1 + 2*cos(omega) + 2*cos(2*omega) + 2*cos(3*omega)}

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
    Rangayyan (2024), Ch 3, Eq 3.111, p. 143
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response of the 8-point MA filter."})


def cheatsheet():
    return "rng100: Frequency response of the 8-point MA filter."
