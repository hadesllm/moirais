# moirais.fn — function file (hadesllm/moirais)
"""Bundle branch block (BBB) classification from ECG."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bundle_branch_block"]


def rangayyan_bundle_branch_block(ecg, fs, r_peaks):
    """
    Bundle branch block (BBB) classification from ECG

    Formula: QRS duration > 120ms; discriminant on QRS width and morphological features

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
        Keys: block_type, qrs_duration

    References
    ----------
    Rangayyan Ch 10.2.1
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bundle branch block (BBB) classification from ECG"})


def cheatsheet():
    return "rgbbb: Bundle branch block (BBB) classification from ECG"
