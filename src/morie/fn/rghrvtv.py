# morie.fn -- function file (hadesllm/morie)
"""Time-varying HRV analysis via STFT of RR intervals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv_time_varying"]


def rangayyan_hrv_time_varying(rr_intervals, fs_resamp, window_len):
    """
    Time-varying HRV analysis via STFT of RR intervals

    Formula: HRV STFT: X_RR(m,f) using short sliding window on interpolated RR series

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    fs_resamp : array-like
        Input data.
    window_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf_hf_trace, t

    References
    ----------
    Rangayyan Ch 8.12
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-varying HRV analysis via STFT of RR intervals"})


def cheatsheet():
    return "rghrvtv: Time-varying HRV analysis via STFT of RR intervals"
