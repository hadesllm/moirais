# morie.fn -- function file (hadesllm/morie)
"""IIR Butterworth filter -- Rangayyan Ch 3."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_iir_filter"]


def rangayyan_iir_filter(x, cutoff, order=4, fs=1.0, btype="low"):
    """Butterworth IIR filter via SOS + zero-phase ``filtfilt``.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float or (float, float)
        Cutoff(s) in Hz.
    order : int
        Filter order (default 4).
    fs : float
        Sampling rate (Hz).
    btype : {"low","high","bandpass","bandstop"}

    Returns
    -------
    RichResult with keys ``signal``, ``sos``, ``order``, ``cutoff``, ``fs``, ``btype``.

    References
    ----------
    Rangayyan Ch 3.
    """
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple, np.ndarray)):
        wn = [float(c) / nyq for c in cutoff]
    else:
        wn = float(cutoff) / nyq
    sos = butter(int(order), wn, btype=btype, output="sos")
    y = sosfiltfilt(sos, x)
    res = RichResult(
        title="Butterworth IIR filter",
        summary_lines=[
            ("Order", int(order)),
            ("Type", btype),
            ("Cutoff (Hz)", cutoff),
            ("Fs (Hz)", float(fs)),
        ],
        interpretation=f"Zero-phase Butterworth {btype} filter, order {order}.",
        payload={
            "signal": y, "sos": sos, "order": int(order),
            "cutoff": cutoff, "fs": float(fs), "btype": btype,
        },
    )
    return with_describe_pointer(res, "rgiir")


# CANONICAL TEST
# >>> fs=100.0; t=np.arange(100)/fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*40*t)
# >>> r = rangayyan_iir_filter(x, cutoff=10, order=4, fs=fs, btype="low")
# >>> r["signal"].shape == x.shape
# True


def cheatsheet():
    return "rgiir: Butterworth IIR filter (zero-phase) -- Rangayyan Ch 3"
