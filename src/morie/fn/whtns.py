"""White Gaussian noise generator."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Judge me by my size, do you?"


def white_noise_gen(N, sigma=1.0, seed=None, **kwargs) -> DescriptiveResult:
    """Generate white Gaussian noise.

    Parameters
    ----------
    N : int
        Number of samples.
    sigma : float
        Standard deviation.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is the noise array; ``extra`` has stats.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=int(N))
    return DescriptiveResult(
        name="white_noise_gen",
        value=noise,
        extra={"sigma": sigma, "n": int(N), "actual_std": float(np.std(noise)), "actual_mean": float(np.mean(noise))},
    )


whtns = white_noise_gen


def cheatsheet() -> str:
    return "white_noise_gen({}) -> White Gaussian noise generator."
