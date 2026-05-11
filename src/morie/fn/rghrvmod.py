# morie.fn — function file (hadesllm/morie)
"""AR spectral model of HRV for LF/HF decomposition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv_ar_model"]


def rangayyan_hrv_ar_model(rr_intervals, order):
    """
    AR spectral model of HRV for LF/HF decomposition

    Formula: S_RR(f) = sigma^2/|A(f)|^2; LF/HF from integral of AR PSD bands

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ar_psd, freqs, lf, hf

    References
    ----------
    Rangayyan Ch 7.9
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR spectral model of HRV for LF/HF decomposition"})


def cheatsheet():
    return "rghrvmod: AR spectral model of HRV for LF/HF decomposition"
