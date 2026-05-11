"""Impulse response of the 8-point MA filter as a sum of shifted deltas.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_impulse_response"]


def rangayyan_ch3_ma_8point_impulse_response(n):
    """
    Impulse response of the 8-point MA filter as a sum of shifted deltas.

    Formula: h(n) = (1/8) * [delta(n) + delta(n-1) + ... + delta(n-7)]

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.109, p. 142
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Impulse response of the 8-point MA filter as a sum of shifted deltas."})


def cheatsheet():
    return "rng098: Impulse response of the 8-point MA filter as a sum of shifted deltas."
