"""Short-Time Fourier Transform analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def stft_analysis(
    x: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> DescriptiveResult:
    """Compute Short-Time Fourier Transform of a signal.

    :param x: 1-D input signal.
    :param window_size: FFT window size (default 256).
    :param hop: Hop size between frames (default 128).
    :param window: Window function name (default 'hann').
    :return: DescriptiveResult with spectrogram S, times, and freqs in extra.
    """
    from moirais._adaptive import stft

    x = np.asarray(x, dtype=float).ravel()
    S, times, freqs = stft(x, window_size=window_size, hop=hop, window=window)
    return DescriptiveResult(
        name="stft",
        value=None,
        extra={"spectrogram": S, "times": times, "freqs": freqs},
    )


stfta = stft_analysis


def cheatsheet() -> str:
    return "stft_analysis({}) -> Short-Time Fourier Transform analysis."
