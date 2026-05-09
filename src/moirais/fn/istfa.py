# moirais.fn — function file (hadesllm/moirais)
"""Inverse Short-Time Fourier Transform signal synthesis."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def istft_synth(
    x: np.ndarray,
    window_size: int = 256,
    hop: int = 128,
    window: str = "hann",
) -> SignalResult:
    """Reconstruct a signal from its STFT (round-trip synthesis).

    :param x: 1-D input signal to transform and reconstruct.
    :param window_size: FFT window size (default 256).
    :param hop: Hop size between frames (default 128).
    :param window: Window function name (default 'hann').
    :return: SignalResult with reconstructed signal.
    """
    from moirais._adaptive import istft, stft

    x = np.asarray(x, dtype=float).ravel()
    S, _, _ = stft(x, window_size=window_size, hop=hop, window=window)
    result = istft(S, window_size=window_size, hop=hop, window=window)
    return SignalResult(name="istft", filtered=result, n_samples=len(result))


istfa = istft_synth


def cheatsheet() -> str:
    return "istft_synth({}) -> Inverse Short-Time Fourier Transform signal synthesis."
