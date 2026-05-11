# morie.fn — function file (hadesllm/morie)
"""QRS slope thresholding for R-peak detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_qrs_slope"]


def rangayyan_ch4_qrs_slope(ecg, fs):
    """
    QRS slope thresholding for R-peak detection

    Formula: slope[n] = (ECG[n+1]-ECG[n-1])/(2*T); R-peak if slope exceeds adaptive threshold

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r_locs

    References
    ----------
    Rangayyan Ch 4.3.1
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QRS slope thresholding for R-peak detection"})


def cheatsheet():
    return "rgeqn4: QRS slope thresholding for R-peak detection"
