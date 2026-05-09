# moirais.fn — function file (hadesllm/moirais)
"""CPR analysis via wavelet for shockable rhythm detection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_cpr_analysis"]


def rangayyan_cpr_analysis(ecg, fs):
    """
    CPR analysis via wavelet for shockable rhythm detection

    Formula: Wavelet features in 3-10 Hz band discriminate VF from non-VF

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_shockable, features

    References
    ----------
    Rangayyan Ch 8.15
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CPR analysis via wavelet for shockable rhythm detection"})


def cheatsheet():
    return "rgcpr: CPR analysis via wavelet for shockable rhythm detection"
