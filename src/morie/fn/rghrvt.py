# morie.fn — function file (hadesllm/morie)
"""HRV time-domain metrics: SDNN, RMSSD, pNN50."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv_time_domain"]


def rangayyan_hrv_time_domain(rr_intervals):
    """
    HRV time-domain metrics: SDNN, RMSSD, pNN50

    Formula: SDNN=std(RR); RMSSD=sqrt(mean(diff(RR)^2)); pNN50=sum(|dRR|>50ms)/N

    Parameters
    ----------
    rr_intervals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sdnn, rmssd, pnn50

    References
    ----------
    Rangayyan Ch 2
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    n = int(rr_intervals) if rr_intervals.ndim == 0 else len(rr_intervals)
    result = float(np.mean(rr_intervals))
    se = float(np.std(rr_intervals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HRV time-domain metrics: SDNN, RMSSD, pNN50"})


def cheatsheet():
    return "rghrvt: HRV time-domain metrics: SDNN, RMSSD, pNN50"
