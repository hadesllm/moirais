# morie.fn -- function file (hadesllm/morie)
"""Short-time Fourier transform -- Rangayyan Ch 4."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_stft"]


def rangayyan_stft(x, fs=1.0, nperseg=256, noverlap=None, window="hann"):
    """Short-time Fourier transform / spectrogram.

    Parameters
    ----------
    x : array-like
    fs : float
    nperseg : int
    noverlap : int, optional
    window : str

    Returns
    -------
    RichResult with keys ``freqs``, ``times``, ``Sxx``, ``nperseg``,
    ``noverlap``, ``fs``.

    References
    ----------
    Rangayyan Ch 4.
    """
    from scipy.signal import spectrogram

    x = np.asarray(x, dtype=float)
    nperseg = min(int(nperseg), x.size)
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = spectrogram(x, fs=fs, window=window, nperseg=nperseg,
                            noverlap=noverlap, scaling="density", mode="psd")
    res = RichResult(
        title="Short-Time Fourier Transform",
        summary_lines=[
            ("nperseg", nperseg),
            ("noverlap", int(noverlap)),
            ("Window", window),
            ("Fs (Hz)", float(fs)),
            ("Frames", int(t.size)),
            ("Freq bins", int(f.size)),
        ],
        interpretation=f"STFT: {t.size} frames × {f.size} freq bins.",
        payload={"freqs": f, "times": t, "Sxx": Sxx,
                 "nperseg": nperseg, "noverlap": int(noverlap), "fs": float(fs)},
    )
    return with_describe_pointer(res, "rgstf")


# CANONICAL TEST
# >>> fs = 100.0
# >>> t = np.arange(1024)/fs
# >>> x = np.sin(2*np.pi*10*t)
# >>> r = rangayyan_stft(x, fs=fs, nperseg=128)
# >>> r["Sxx"].shape[0] == r["freqs"].size
# True


def cheatsheet():
    return "rgstf: short-time Fourier transform -- Rangayyan Ch 4"
