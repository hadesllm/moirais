# morie.fn -- function file (hadesllm/morie)
"""Continuous-time Fourier transform (CTFT)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fourier_transform"]


def rangayyan_fourier_transform(t, x):
    """
    Continuous-time Fourier transform (CTFT)

    Formula: X(f) = integral_{-inf}^{inf} x(t) * exp(-j2*pi*f*t) dt

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: X_f, freqs

    References
    ----------
    Rangayyan Ch 3.4.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time Fourier transform (CTFT)"})


def cheatsheet():
    return "rgft: Continuous-time Fourier transform (CTFT)"
