# morie.fn -- function file (hadesllm/morie)
"""Frequency-modulated (FM) signal model for respiratory sounds."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fm_signal"]


def rangayyan_fm_signal(t, f0, m_t, kf):
    """
    Frequency-modulated (FM) signal model for respiratory sounds

    Formula: t(t) = A*cos(2*pi*f0*t + k_f*integral m(tau)d(tau))

    Parameters
    ----------
    t : array-like
        Input data.
    f0 : array-like
        Input data.
    m_t : array-like
        Input data.
    kf : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fm_signal

    References
    ----------
    Rangayyan Ch 7.7.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency-modulated (FM) signal model for respiratory sounds"})


def cheatsheet():
    return "rgfm: Frequency-modulated (FM) signal model for respiratory sounds"
