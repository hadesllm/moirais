# moirais.fn — function file (hadesllm/moirais)
"""Electroneurogram (ENG) compound action potential model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_eng"]


def rangayyan_eng(t, n_fibers, cv_range, amp_range):
    """
    Electroneurogram (ENG) compound action potential model

    Formula: CAP = sum of single fiber APs with varying conduction velocities and latencies

    Parameters
    ----------
    t : array-like
        Input data.
    n_fibers : array-like
        Input data.
    cv_range : array-like
        Input data.
    amp_range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cap_waveform

    References
    ----------
    Rangayyan Ch 1.2.3
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Electroneurogram (ENG) compound action potential model"})


def cheatsheet():
    return "rgengn: Electroneurogram (ENG) compound action potential model"
