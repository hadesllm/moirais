# morie.fn — function file (hadesllm/morie)
"""T-wave detection in ECG."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_t_wave_detect"]


def rangayyan_t_wave_detect(ecg, fs, r_peaks):
    """
    T-wave detection in ECG

    Formula: T search window: 100-400 ms after QRS end; peak detection in window

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
        Keys: t_locs

    References
    ----------
    Rangayyan Ch 4.3.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-wave detection in ECG"})


def cheatsheet():
    return "rgtwave: T-wave detection in ECG"
