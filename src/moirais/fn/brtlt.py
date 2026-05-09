# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bartlett power spectral density estimate."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def bartlett_psd_fn(x: np.ndarray, fs: float = 1.0, n_segments: int = 8) -> DescriptiveResult:
    """Bartlett averaged periodogram PSD estimate.

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :param n_segments: Number of non-overlapping segments (default 8).
    :return: DescriptiveResult with freqs and psd in extra.
    """
    from moirais._spectral import bartlett_psd

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = bartlett_psd(x, fs=fs, n_segments=n_segments)
    return DescriptiveResult(name="bartlett_psd", value=None, extra={"freqs": freqs, "psd": psd})


brtlt = bartlett_psd_fn


def cheatsheet() -> str:
    return "bartlett_psd_fn({}) -> Bartlett power spectral density estimate."
