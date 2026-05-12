# morie.fn -- function file (hadesllm/morie)
"""Chebyshev type I IIR filter.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

__all__ = ['chflt']

_QUOTE = "The ripple is strong with this one. --"


def chflt(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    cutoff: float | tuple[float, float] = 100.0,
    order: int = 4,
    btype: str = "low",
    rp: float = 1.0,
) -> SignalResult:
    """Chebyshev type I filter design and application.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    cutoff : float or tuple
        Cutoff frequency (Hz). Tuple for bandpass/bandstop.
    order : int
        Filter order.
    btype : str
        ``'low'``, ``'high'``, ``'band'``, or ``'bandstop'``.
    rp : float
        Maximum ripple in the passband (dB).

    Returns
    -------
    SignalResult
    """
    from scipy.signal import cheby1, sosfiltfilt

    x = np.asarray(x, dtype=float).ravel()
    nyq = fs / 2.0
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    wn = np.clip(wn, 1e-6, 1.0 - 1e-6)

    sos = cheby1(order, rp, wn, btype=btype, output="sos")
    y = sosfiltfilt(sos, x)

    return SignalResult(
        name="chflt",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"order": order, "cutoff": cutoff, "btype": btype, "rp": rp},
    )


def cheatsheet() -> str:
    return "chflt({}) -> Chebyshev type I filter."
