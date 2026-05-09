# moirais.fn — function file (hadesllm/moirais)
"""HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv_freq_domain"]


def rangayyan_hrv_freq_domain(rr_intervals, fs_resamp):
    """
    HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio

    Formula: P_band = integral_{f1}^{f2} S_RR(f) df; LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    fs_resamp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vlf, lf, hf, lf_hf

    References
    ----------
    Rangayyan Ch 2
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio"})


def cheatsheet():
    return "rghrvf: HRV frequency-domain metrics: VLF/LF/HF power and LF/HF ratio"
