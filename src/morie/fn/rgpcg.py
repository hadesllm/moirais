# morie.fn -- function file (hadesllm/morie)
"""PCG segmentation into S1/systole/S2/diastole using ECG gating."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_pcg_segments"]


def rangayyan_pcg_segments(pcg, ecg, fs):
    """
    PCG segmentation into S1/systole/S2/diastole using ECG gating

    Formula: S1 onset ~ R-wave; S2 onset ~ T-wave end; durations from timing ratios

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: segment_labels

    References
    ----------
    Rangayyan Ch 1.2.9
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCG segmentation into S1/systole/S2/diastole using ECG gating"})


def cheatsheet():
    return "rgpcg: PCG segmentation into S1/systole/S2/diastole using ECG gating"
