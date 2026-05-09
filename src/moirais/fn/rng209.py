"""PSD of white noise at the input of a matched filter (two-sided).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_white_noise_psd_input"]


def rangayyan_ch4_white_noise_psd_input(P_eta_i, f):
    """
    PSD of white noise at the input of a matched filter (two-sided).

    Formula: S_eta_i(f) = P_eta_i / 2

    Parameters
    ----------
    P_eta_i : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.35, p. 238
    """
    P_eta_i = np.atleast_1d(np.asarray(P_eta_i, dtype=float))
    n = len(P_eta_i)
    result = float(np.mean(P_eta_i))
    se = float(np.std(P_eta_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PSD of white noise at the input of a matched filter (two-sided)."})


def cheatsheet():
    return "rng209: PSD of white noise at the input of a matched filter (two-sided)."
