# morie.fn — function file (hadesllm/morie)
"""Signal derivative in Hz domain."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "He who has a why to live can bear almost any how. — Friedrich Nietzsche"


def derivative_hz(x, fs=1.0, order=1, **kwargs) -> DescriptiveResult:
    """Compute signal derivative via frequency domain differentiation.

    Multiplies FFT by (j*2*pi*f)^order, then inverse FFT.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency.
    order : int
        Derivative order (default 1).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    deriv_filter = (1j * 2 * np.pi * freqs) ** order
    Y = X * deriv_filter
    y = np.real(np.fft.ifft(Y))
    return DescriptiveResult(
        name="derivative_hz",
        value=float(np.max(np.abs(y))),
        extra={
            "order": order,
            "fs": fs,
            "max_abs": float(np.max(np.abs(y))),
            "rms": float(np.sqrt(np.mean(y**2))),
        },
    )


drvhz = derivative_hz


def cheatsheet() -> str:
    return "derivative_hz({}) -> Signal derivative in Hz domain."
