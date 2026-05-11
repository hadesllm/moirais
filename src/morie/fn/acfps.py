# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""ACF estimated from power spectral density via inverse FFT."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def acf_from_psd_fn(x: np.ndarray, fs: float = 1.0) -> DescriptiveResult:
    """Compute ACF from PSD via inverse FFT (Wiener-Khinchin theorem).

    :param x: 1-D input signal.
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with acf array in extra.
    """
    from morie._spectral import acf_from_psd, periodogram

    x = np.asarray(x, dtype=float).ravel()
    freqs, psd = periodogram(x, fs=fs)
    acf = acf_from_psd(psd)
    return DescriptiveResult(name="acf_from_psd", value=None, extra={"acf": acf})


acfps = acf_from_psd_fn


def cheatsheet() -> str:
    return "acf_from_psd_fn({}) -> ACF estimated from power spectral density via inverse FFT."
