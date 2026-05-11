# morie.fn — function file (hadesllm/morie)
"""Colored (1/f^alpha) noise generator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def colored_noise_gen(N, alpha=1.0, fs=1.0, seed=None, **kwargs) -> DescriptiveResult:
    """Generate colored noise with power spectral density proportional
    to :math:`1/f^{\\alpha}`.

    Parameters
    ----------
    N : int
        Number of samples.
    alpha : float
        Spectral exponent. 0 = white, 1 = pink, 2 = brown/red.
    fs : float
        Sampling frequency (Hz).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is the noise array.
    """
    N = int(N)
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, size=N)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    freqs[0] = 1.0
    X = X / (freqs ** (alpha / 2.0))
    noise = np.fft.irfft(X, n=N)
    noise = noise / np.std(noise) if np.std(noise) > 0 else noise
    return DescriptiveResult(
        name="colored_noise_gen",
        value=noise,
        extra={"alpha": alpha, "fs": fs, "n": N},
    )


clrns = colored_noise_gen


def cheatsheet() -> str:
    return "colored_noise_gen({}) -> Colored (1/f^alpha) noise generator."
