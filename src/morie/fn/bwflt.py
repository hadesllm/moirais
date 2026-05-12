# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Butterworth IIR filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Real knowledge is to know the extent of one's ignorance. -- Confucius"


def butterworth_filter(x, cutoff, fs, order: int = 4, btype: str = "low") -> SignalResult:
    r"""Apply a Butterworth IIR filter to signal *x*.

    .. math::

        |H(j\\omega)|^2 = \\frac{1}{1 + (\\omega / \\omega_c)^{2N}}

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float or tuple
        Cutoff frequency (Hz). For bandpass/bandstop, a 2-element sequence.
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order. Default 4.
    btype : str
        Filter type: 'low', 'high', 'bandpass', 'bandstop'. Default 'low'.

    Returns
    -------
    SignalResult
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        Wn = [float(c) / nyq for c in cutoff]
    else:
        Wn = float(cutoff) / nyq
    sos = butter(order, Wn, btype=btype, output="sos")
    y = sosfiltfilt(sos, x)
    return SignalResult(
        name="butterworth_filter",
        filtered=y,
        fs=fs,
        n_samples=len(x),
        extra={"order": order, "cutoff": cutoff, "btype": btype},
    )


bwflt = butterworth_filter


def cheatsheet() -> str:
    return "butterworth_filter({}) -> Butterworth IIR filter."
