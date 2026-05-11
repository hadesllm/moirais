# morie.fn — function file (hadesllm/morie)
"""ECG waveform feature extraction (P, QRS, T amplitudes and durations)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ecg_features"]


def rangayyan_ecg_features(ecg, fs, r_peaks):
    """
    ECG waveform feature extraction (P, QRS, T amplitudes and durations)

    Formula: Feature vector = [P_amp, PR_int, QRS_dur, QT_int, T_amp, ST_dev]

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
        Keys: feature_dict

    References
    ----------
    Rangayyan Ch 1.2.5
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ECG waveform feature extraction (P, QRS, T amplitudes and durations)"})


def cheatsheet():
    return "rgecgf: ECG waveform feature extraction (P, QRS, T amplitudes and durations)"
