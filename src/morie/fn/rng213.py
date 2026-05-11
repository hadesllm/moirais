"""Peak-power SNR at output of a matched filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_peak_power_snr"]


def rangayyan_ch4_peak_power_snr(M_y, P_eta_o):
    """
    Peak-power SNR at output of a matched filter.

    Formula: M_y^2 / P_eta_o = instantaneous_peak_power_of_signal / noise_mean_power

    Parameters
    ----------
    M_y : array-like
        Input data.
    P_eta_o : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.39, p. 238
    """
    M_y = np.atleast_1d(np.asarray(M_y, dtype=float))
    n = len(M_y)
    result = float(np.mean(M_y))
    se = float(np.std(M_y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peak-power SNR at output of a matched filter."})


def cheatsheet():
    return "rng213: Peak-power SNR at output of a matched filter."
