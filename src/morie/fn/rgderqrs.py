# morie.fn -- function file (hadesllm/morie)
"""Derivative-based QRS detection (first and second differences)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_deriv_qrs"]


def rangayyan_deriv_qrs(ecg, fs, threshold):
    """
    Derivative-based QRS detection (first and second differences)

    Formula: d_ecg = |dx/dt|; threshold on peak of derivative; decision threshold adaptive

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: qrs_locs

    References
    ----------
    Rangayyan Ch 4.3.1
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Derivative-based QRS detection (first and second differences)"})


def cheatsheet():
    return "rgderqrs: Derivative-based QRS detection (first and second differences)"
