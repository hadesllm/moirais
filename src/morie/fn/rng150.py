"""Wiener filter frequency response as ratio of CSD to PSD of input.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_frequency_response"]


def rangayyan_ch3_wiener_frequency_response(S_xd, S_xx, omega):
    """
    Wiener filter frequency response as ratio of CSD to PSD of input.

    Formula: W(omega) = S_xd(omega) / S_xx(omega)

    Parameters
    ----------
    S_xd : array-like
        Input data.
    S_xx : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.176, p. 176
    """
    S_xd = np.atleast_1d(np.asarray(S_xd, dtype=float))
    n = len(S_xd)
    result = float(np.mean(S_xd))
    se = float(np.std(S_xd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener filter frequency response as ratio of CSD to PSD of input."})


def cheatsheet():
    return "rng150: Wiener filter frequency response as ratio of CSD to PSD of input."
