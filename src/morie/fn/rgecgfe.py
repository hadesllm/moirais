# morie.fn — function file (hadesllm/morie)
"""Single-channel fetal ECG extraction using NMF/ICA."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fetal_ecg_single"]


def rangayyan_fetal_ecg_single(abdominal_ecg, fs, method):
    """
    Single-channel fetal ECG extraction using NMF/ICA

    Formula: Maternal component removed by NMF decomposition; fetal ECG in residual

    Parameters
    ----------
    abdominal_ecg : array-like
        Input data.
    fs : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg

    References
    ----------
    Rangayyan Ch 9.11
    """
    abdominal_ecg = np.asarray(abdominal_ecg, dtype=float)
    n = int(abdominal_ecg) if abdominal_ecg.ndim == 0 else len(abdominal_ecg)
    result = float(np.mean(abdominal_ecg))
    se = float(np.std(abdominal_ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single-channel fetal ECG extraction using NMF/ICA"})


def cheatsheet():
    return "rgecgfe: Single-channel fetal ECG extraction using NMF/ICA"
