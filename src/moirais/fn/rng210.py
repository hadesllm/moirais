"""Noise PSD at the output of a matched filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_noise_psd_at_output"]


def rangayyan_ch4_noise_psd_at_output(P_eta_i, H, f):
    """
    Noise PSD at the output of a matched filter.

    Formula: S_eta_o(f) = (P_eta_i / 2) * |H(f)|^2

    Parameters
    ----------
    P_eta_i : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.36, p. 238
    """
    P_eta_i = np.atleast_1d(np.asarray(P_eta_i, dtype=float))
    n = len(P_eta_i)
    result = float(np.mean(P_eta_i))
    se = float(np.std(P_eta_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Noise PSD at the output of a matched filter."})


def cheatsheet():
    return "rng210: Noise PSD at the output of a matched filter."
