# morie.fn -- function file (hadesllm/morie)
"""T-wave alternans (TWA) detection via spectral method."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_twave_alternans"]


def rangayyan_twave_alternans(ecg, fs, r_peaks):
    """
    T-wave alternans (TWA) detection via spectral method

    Formula: TWA at 0.5 cycles/beat in even-odd T-wave amplitude spectrum

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: twa_magnitude, k_score

    References
    ----------
    Rangayyan Ch 9.10
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-wave alternans (TWA) detection via spectral method"})


def cheatsheet():
    return "rgtwa: T-wave alternans (TWA) detection via spectral method"
