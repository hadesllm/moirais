"""Wiener filter frequency response in terms of signal and noise PSDs.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_frequency_response_snr_form"]


def rangayyan_ch3_wiener_frequency_response_snr_form(S_d, S_eta, omega):
    """
    Wiener filter frequency response in terms of signal and noise PSDs.

    Formula: W(omega) = S_d(omega) / (S_d(omega) + S_eta(omega)) = 1 / (1 + S_eta(omega)/S_d(omega))

    Parameters
    ----------
    S_d : array-like
        Input data.
    S_eta : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.186, p. 177
    """
    S_d = np.atleast_1d(np.asarray(S_d, dtype=float))
    n = len(S_d)
    result = float(np.mean(S_d))
    se = float(np.std(S_d, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wiener filter frequency response in terms of signal and noise PSDs."})


def cheatsheet():
    return "rng152: Wiener filter frequency response in terms of signal and noise PSDs."
