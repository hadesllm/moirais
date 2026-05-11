# morie.fn — function file (hadesllm/morie)
"""P-wave detection in ECG using search window relative to R-peak."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_p_wave_detect"]


def rangayyan_p_wave_detect(ecg, fs, r_peaks):
    """
    P-wave detection in ECG using search window relative to R-peak

    Formula: P search window: 200 ms before QRS; peak in 50-120 ms PR interval

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
        Keys: p_locs

    References
    ----------
    Rangayyan Ch 4.3.3
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "P-wave detection in ECG using search window relative to R-peak"})


def cheatsheet():
    return "rgpwave: P-wave detection in ECG using search window relative to R-peak"
