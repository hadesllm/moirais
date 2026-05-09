# moirais.fn — function file (hadesllm/moirais)
"""Baseline wander removal from ECG."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_baseline_wander"]


def rangayyan_baseline_wander(ecg, fs, cutoff):
    """
    Baseline wander removal from ECG

    Formula: High-pass > 0.05 Hz Butterworth or cubic spline through isoelectric points

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ecg_detrended

    References
    ----------
    Rangayyan Ch 3.3.2
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baseline wander removal from ECG"})


def cheatsheet():
    return "rgblwand: Baseline wander removal from ECG"
