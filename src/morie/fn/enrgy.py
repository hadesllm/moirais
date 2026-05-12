# morie.fn -- function file (hadesllm/morie)
"""Energy spectral density."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your focus determines your reality."


def energy_density(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute the energy spectral density |X(f)|^2.

    .. math::

        E(f) = |X(f)|^2

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.rfft(x)
    esd = np.abs(X) ** 2 / fs
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    total_energy = float(np.sum(x**2) / fs)
    return DescriptiveResult(
        name="energy_density",
        value=total_energy,
        extra={"esd": esd, "freqs": freqs, "total_energy": total_energy, "fs": fs},
    )


enrgy = energy_density


def cheatsheet() -> str:
    return "energy_density({}) -> Energy spectral density."
