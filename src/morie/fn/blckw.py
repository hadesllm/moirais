# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""SNR degradation model. 'All communications jammed.' -- Blackout"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def snr_degradation(
    signal: np.ndarray,
    *,
    snr_db: float = 20.0,
    noise_type: str = "gaussian",
    seed: int = 42,
) -> DescriptiveResult:
    """Model signal-to-noise ratio degradation by adding calibrated noise.

    Parameters
    ----------
    signal : array-like
        Clean input signal.
    snr_db : float
        Target SNR in decibels.
    noise_type : str
        Noise type: 'gaussian', 'uniform', or 'impulsive'.
    seed : int
        Random seed.

    Returns
    -------
    DescriptiveResult
        With ``value`` = noisy signal and ``extra`` containing
        actual SNR and noise power.
    """
    x = np.asarray(signal, dtype=float).ravel()
    if len(x) == 0:
        raise ValueError("Signal must be non-empty")

    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(x**2))
    if sig_power < 1e-30:
        sig_power = 1e-30

    noise_power = sig_power / (10 ** (snr_db / 10))

    if noise_type == "gaussian":
        noise = rng.normal(0, np.sqrt(noise_power), len(x))
    elif noise_type == "uniform":
        amp = np.sqrt(3 * noise_power)
        noise = rng.uniform(-amp, amp, len(x))
    elif noise_type == "impulsive":
        noise = rng.normal(0, np.sqrt(noise_power), len(x))
        mask = rng.random(len(x)) < 0.05
        noise[mask] *= 10
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    noisy = x + noise
    actual_snr = 10 * np.log10(sig_power / max(np.mean(noise**2), 1e-30))

    return DescriptiveResult(
        name="snr_degradation",
        value=noisy,
        extra={
            "target_snr_db": snr_db,
            "actual_snr_db": float(actual_snr),
            "noise_power": float(np.mean(noise**2)),
            "noise_type": noise_type,
            "n": len(x),
        },
    )


blckw = snr_degradation


def cheatsheet() -> str:
    return "snr_degradation({}) -> SNR degradation model. 'All communications jammed.' -- Black"
