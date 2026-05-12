# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""AR model power spectral density from estimated coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def ar_spectrum_fn(
    x: np.ndarray,
    order: int = 10,
    fs: float = 1.0,
    n_points: int = 512,
) -> DescriptiveResult:
    """Compute AR model PSD from Yule-Walker estimated coefficients.

    :param x: 1-D input signal.
    :param order: AR model order (default 10).
    :param fs: Sampling frequency (default 1.0).
    :param n_points: Number of frequency points (default 512).
    :return: DescriptiveResult with freqs and psd in extra.
    """
    from morie._armodel import ar_spectrum, ar_yule_walker

    x = np.asarray(x, dtype=float).ravel()
    a, sigma2 = ar_yule_walker(x, order=order)
    freqs, psd = ar_spectrum(a, sigma2, fs=fs, n_points=n_points)
    return DescriptiveResult(
        name="ar_spectrum",
        value=None,
        extra={"freqs": freqs, "psd": psd},
    )


arspc = ar_spectrum_fn


def cheatsheet() -> str:
    return "ar_spectrum_fn({}) -> AR model power spectral density from estimated coefficients."
