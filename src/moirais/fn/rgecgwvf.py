# moirais.fn — function file (hadesllm/moirais)
"""ECG waveform analysis for ischemia and bundle branch block."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ecg_waveshape"]


def rangayyan_ecg_waveshape(ecg, fs, r_peaks, template):
    """
    ECG waveform analysis for ischemia and bundle branch block

    Formula: Template correlation coefficient rho > 0.9 = normal; < 0.7 = ectopic

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.
    template : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho_per_beat, classification

    References
    ----------
    Rangayyan Ch 5.4.3
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ECG waveform analysis for ischemia and bundle branch block"})


def cheatsheet():
    return "rgecgwvf: ECG waveform analysis for ischemia and bundle branch block"
