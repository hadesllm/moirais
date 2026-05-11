"""Spectrogram (squared magnitude STFT)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectrogram_fn(
    x: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> DescriptiveResult:
    """Compute power spectrogram of a signal.

    :param x: 1-D input signal.
    :param window_size: FFT window size (default 256).
    :param hop: Hop size between frames (default 128).
    :param window: Window function name (default 'hann').
    :return: DescriptiveResult with power, times, freqs in extra.
    """
    from morie._adaptive import spectrogram

    x = np.asarray(x, dtype=float).ravel()
    power, times, freqs = spectrogram(x, window_size=window_size, hop=hop, window=window)
    return DescriptiveResult(
        name="spectrogram",
        value=None,
        extra={"power": power, "times": times, "freqs": freqs},
    )


spgrm = spectrogram_fn


def cheatsheet() -> str:
    return "spectrogram_fn({}) -> Spectrogram (squared magnitude STFT)."
