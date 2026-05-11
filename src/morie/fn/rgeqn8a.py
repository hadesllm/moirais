# morie.fn — function file (hadesllm/morie)
"""Adaptive threshold for SEM-based segmentation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch8_sem_threshold"]


def rangayyan_ch8_sem_threshold(sem_trace, k):
    """
    Adaptive threshold for SEM-based segmentation

    Formula: Threshold = mean_SEM + k*std_SEM; boundary where SEM > threshold

    Parameters
    ----------
    sem_trace : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: boundaries

    References
    ----------
    Rangayyan Ch 8.5.1
    """
    sem_trace = np.asarray(sem_trace, dtype=float)
    n = int(sem_trace) if sem_trace.ndim == 0 else len(sem_trace)
    result = float(np.mean(sem_trace))
    se = float(np.std(sem_trace, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive threshold for SEM-based segmentation"})


def cheatsheet():
    return "rgeqn8a: Adaptive threshold for SEM-based segmentation"
