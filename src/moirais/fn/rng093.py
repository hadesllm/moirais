"""Frequency response of the Hann filter on the unit circle.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_frequency_response_raw"]


def rangayyan_ch3_hann_frequency_response_raw(omega):
    """
    Frequency response of the Hann filter on the unit circle.

    Formula: H(omega) = H(z)|_{z=exp(j*omega)} = (1/4) * [1 + 2*exp(-j*omega) + exp(-j*2*omega)]

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
    Rangayyan (2024), Ch 3, Eq 3.104, p. 141
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency response of the Hann filter on the unit circle."})


def cheatsheet():
    return "rng093: Frequency response of the Hann filter on the unit circle."
