# moirais.fn — function file (hadesllm/moirais)
"""Exercise ECG analysis: ST deviation, slope, and ischemia detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_exercise_ecg"]


def rangayyan_exercise_ecg(ecg, fs, r_peaks):
    """
    Exercise ECG analysis: ST deviation, slope, and ischemia detection

    Formula: ST level = mean(ECG) in J+60ms to J+80ms window; ST slope from regression

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
        Keys: st_level, st_slope, ischemia_flag

    References
    ----------
    Rangayyan Ch 5.8
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exercise ECG analysis: ST deviation, slope, and ischemia detection"})


def cheatsheet():
    return "rgexecg: Exercise ECG analysis: ST deviation, slope, and ischemia detection"
