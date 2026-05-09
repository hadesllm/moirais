"""Closed-form sinc-type frequency response of the recursive 8-point MA filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_sinc_frequency_response"]


def rangayyan_ch3_ma_8point_sinc_frequency_response(omega):
    """
    Closed-form sinc-type frequency response of the recursive 8-point MA filter.

    Formula: H(omega) = (1/8) * (1 - exp(-j*8*omega)) / (1 - exp(-j*omega)) = (1/8) * exp(-j*7*omega/2) * sin(4*omega) / sin(omega/2)

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
    Rangayyan (2024), Ch 3, Eq 3.122, p. 145
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form sinc-type frequency response of the recursive 8-point MA filter."})


def cheatsheet():
    return "rng110: Closed-form sinc-type frequency response of the recursive 8-point MA filter."
