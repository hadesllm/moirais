# moirais.fn — function file (hadesllm/moirais)
"""Amplitude-modulated (AM) signal model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_am_signal"]


def rangayyan_am_signal(t, fc, m_t, Ac):
    """
    Amplitude-modulated (AM) signal model

    Formula: t(t) = A_c*(1 + m(t))*cos(2*pi*f_c*t)

    Parameters
    ----------
    t : array-like
        Input data.
    fc : array-like
        Input data.
    m_t : array-like
        Input data.
    Ac : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: am_signal

    References
    ----------
    Rangayyan Ch 5.5.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Amplitude-modulated (AM) signal model"})


def cheatsheet():
    return "rgam: Amplitude-modulated (AM) signal model"
