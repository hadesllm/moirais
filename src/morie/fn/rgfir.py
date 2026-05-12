# morie.fn — function file (hadesllm/morie)
"""FIR filter design (windowed sinc) — Rangayyan Ch 3."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_fir_filter"]


def rangayyan_fir_filter(x, cutoff, order=51, fs=1.0, window="hamming"):
    """Windowed-sinc FIR lowpass filter.

    Designs a linear-phase FIR lowpass filter of length ``order`` using
    the windowed-sinc method::

        h[n] = w[n] * 2*fc * sinc(2*fc * (n - M/2))

    with ``fc = cutoff / fs`` and applies it to ``x`` via zero-phase
    forward-backward convolution.

    Parameters
    ----------
    x : array-like
        Input signal.
    cutoff : float
        Cutoff frequency (Hz if ``fs`` given, else normalised to Nyquist=0.5).
    order : int
        Number of taps (odd recommended). Default 51.
    fs : float
        Sampling frequency (Hz). Default 1.0.
    window : str
        Window function name (``hamming``, ``hann``, ``blackman``, ``rect``).

    Returns
    -------
    RichResult with keys ``signal``, ``taps``, ``order``, ``cutoff``, ``fs``.

    References
    ----------
    Rangayyan, R.M. *Biomedical Signal Analysis* (Wiley/IEEE, 3rd ed., 2015), Ch. 3.
    """
    from scipy.signal import firwin, filtfilt

    x = np.asarray(x, dtype=float)
    order = int(order)
    if order < 3:
        order = 3
    if order % 2 == 0:
        order += 1  # ensure odd (linear-phase Type I)
    nyq = 0.5 * fs
    fc = cutoff / nyq
    fc = float(np.clip(fc, 1e-6, 1 - 1e-6))
    taps = firwin(order, fc, window=window)
    # filtfilt needs len(x) > 3 * order; fall back to single-pass for shorts.
    padlen = 3 * order
    if x.size > padlen:
        y = filtfilt(taps, [1.0], x)
    else:
        from scipy.signal import lfilter
        y = lfilter(taps, [1.0], x)
    res = RichResult(
        title="FIR lowpass filter (windowed sinc)",
        summary_lines=[
            ("Order", order),
            ("Cutoff (Hz)", float(cutoff)),
            ("Fs (Hz)", float(fs)),
            ("Window", window),
            ("Output length", int(y.size)),
        ],
        interpretation=(
            f"Zero-phase FIR lowpass of order {order} with cutoff "
            f"{cutoff:.4g} Hz applied to {x.size} samples."
        ),
        payload={
            "signal": y,
            "taps": taps,
            "order": order,
            "cutoff": float(cutoff),
            "fs": float(fs),
            "window": window,
        },
    )
    return with_describe_pointer(res, "rgfir")


# CANONICAL TEST
# >>> import numpy as np
# >>> fs = 100.0
# >>> t = np.arange(100) / fs
# >>> x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*30*t)
# >>> r = rangayyan_fir_filter(x, cutoff=10, order=51, fs=fs)
# >>> r["signal"].shape == x.shape
# True


def cheatsheet():
    return "rgfir: FIR lowpass filter (windowed sinc) — Rangayyan Ch 3"
