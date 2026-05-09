# moirais.fn — function file (hadesllm/moirais)
"""Bartlett method PSD estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def bartlett_psd(x, nseg: int = 8, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Estimate PSD using Bartlett's method (averaged periodograms).

    The signal is divided into *nseg* non-overlapping segments and
    their periodograms are averaged.

    Parameters
    ----------
    x : array-like
        Input signal.
    nseg : int
        Number of segments.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    seg_len = N // nseg
    if seg_len < 2:
        raise ValueError("Signal too short for requested nseg")
    psd_sum = None
    for i in range(nseg):
        seg = x[i * seg_len : (i + 1) * seg_len]
        X = np.fft.rfft(seg)
        psd_seg = (np.abs(X) ** 2) / (seg_len * fs)
        if psd_sum is None:
            psd_sum = psd_seg
        else:
            psd_sum = psd_sum + psd_seg
    psd_avg = psd_sum / nseg
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    return DescriptiveResult(
        name="bartlett_psd",
        value=float(np.max(psd_avg)),
        extra={"psd": psd_avg, "freqs": freqs, "nseg": nseg, "seg_len": seg_len},
    )


psdbt = bartlett_psd


def cheatsheet() -> str:
    return "bartlett_psd({}) -> Bartlett method PSD estimation."
