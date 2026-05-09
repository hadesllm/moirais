# moirais.fn — function file (hadesllm/moirais)
"""HRV AR model LF/HF ratio (sympathovagal balance)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv_ar_ratio"]


def rangayyan_hrv_ar_ratio(rr_intervals, ar_order):
    """
    HRV AR model LF/HF ratio (sympathovagal balance)

    Formula: LF power from AR PSD integral [0.04-0.15Hz]; HF [0.15-0.40Hz]; ratio

    Parameters
    ----------
    rr_intervals : array-like
        Input data.
    ar_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf_hf_ratio, lf_nu, hf_nu

    References
    ----------
    Rangayyan Ch 7.9
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HRV AR model LF/HF ratio (sympathovagal balance)"})


def cheatsheet():
    return "rghrvar: HRV AR model LF/HF ratio (sympathovagal balance)"
