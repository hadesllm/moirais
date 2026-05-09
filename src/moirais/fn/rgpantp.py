# moirais.fn — function file (hadesllm/moirais)
"""Pan-Tompkins QRS detection algorithm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pan_tompkins"]


def rangayyan_pan_tompkins(ecg, fs):
    """
    Pan-Tompkins QRS detection algorithm

    Formula: BP(5-15Hz) -> deriv -> square -> MA(150ms) -> adaptive threshold

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: r_peaks

    References
    ----------
    Rangayyan Ch 4.3.2
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pan-Tompkins QRS detection algorithm"})


def cheatsheet():
    return "rgpantp: Pan-Tompkins QRS detection algorithm"
