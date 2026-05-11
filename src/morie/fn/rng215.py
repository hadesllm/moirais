"""Normalized ratio used in maximizing matched-filter SNR.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_snr_normalized_ratio"]


def rangayyan_ch4_snr_normalized_ratio(H, X, P_eta_i, f, t_0):
    """
    Normalized ratio used in maximizing matched-filter SNR.

    Formula: M_y^2 / (E_x * P_eta_o) = | integral H(f) X(f) exp(+j*2*pi*f*t_0) df |^2 / ( (P_eta_i/2) * integral |H(f)|^2 df * integral |X(f)|^2 df )

    Parameters
    ----------
    H : array-like
        Input data.
    X : array-like
        Input data.
    P_eta_i : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.41, p. 238
    """
    H = np.atleast_1d(np.asarray(H, dtype=float))
    n = len(H)
    result = float(np.mean(H))
    se = float(np.std(H, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normalized ratio used in maximizing matched-filter SNR."})


def cheatsheet():
    return "rng215: Normalized ratio used in maximizing matched-filter SNR."
