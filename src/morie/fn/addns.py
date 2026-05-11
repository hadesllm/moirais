# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Add noise at specified SNR."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun. If you only believe in it when you can see it, you'll never make it through the night."


def add_noise(signal, snr_db=10.0, seed=None, **kwargs) -> DescriptiveResult:
    """Add white Gaussian noise to *signal* at a specified SNR.

    Parameters
    ----------
    signal : array-like
        Clean input signal.
    snr_db : float
        Desired signal-to-noise ratio in dB.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is the noisy signal array.
    """
    signal = np.asarray(signal, dtype=float)
    rng = np.random.default_rng(seed)
    ps = float(np.mean(signal**2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    pn = ps / snr_linear if snr_linear > 0 else 0.0
    noise = rng.normal(0.0, np.sqrt(pn), size=len(signal))
    noisy = signal + noise
    return DescriptiveResult(
        name="add_noise",
        value=noisy,
        extra={"snr_db": snr_db, "noise_power": float(pn), "signal_power": ps, "n": len(signal)},
    )


addns = add_noise


def cheatsheet() -> str:
    return "add_noise({}) -> Add noise at specified SNR."
