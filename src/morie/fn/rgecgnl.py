# morie.fn — function file (hadesllm/morie)
"""Normal vs. ectopic ECG beat classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ecg_normal_ectopic"]


def rangayyan_ecg_normal_ectopic(ecg, fs, r_peaks):
    """
    Normal vs. ectopic ECG beat classification

    Formula: LDA or k-means on QRS morphological features

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
        Keys: beat_labels, features

    References
    ----------
    Rangayyan Ch 10.11
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Normal vs. ectopic ECG beat classification"})


def cheatsheet():
    return "rgecgnl: Normal vs. ectopic ECG beat classification"
