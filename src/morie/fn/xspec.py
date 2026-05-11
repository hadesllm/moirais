"""Cross-spectral density between two signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def cross_spectral_density(x, y, fs: float = 1.0, nperseg: int = 256) -> DescriptiveResult:
    """Compute cross-spectral density between *x* and *y*.

    Parameters
    ----------
    x, y : array-like
        Input signals.
    fs : float
        Sampling frequency. Default 1.0.
    nperseg : int
        Segment length for Welch method. Default 256.

    Returns
    -------
    DescriptiveResult
    """
    from morie._detection import cross_spectral_density as _csd

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f, Pxy = _csd(x, y, fs=fs, nperseg=nperseg)
    return DescriptiveResult(
        name="cross_spectral_density",
        value=len(f),
        extra={"frequencies": f, "csd": Pxy},
    )


xspec = cross_spectral_density


def cheatsheet() -> str:
    return "cross_spectral_density({}) -> Cross-spectral density between two signals."
