# morie.fn — function file (hadesllm/morie)
"""Noise power spectral density."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what we repeatedly do. Excellence is not an act, but a habit. — Aristotle"


def noise_psd(x, fs=1.0, **kwargs) -> DescriptiveResult:
    r"""Estimate the noise power spectral density of *x*.

    For white Gaussian noise the PSD is flat at :math:`\\sigma^2 / f_s`.

    Parameters
    ----------
    x : array-like
        Noise signal.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    var = float(np.var(x, ddof=0))
    psd = var / fs
    return DescriptiveResult(
        name="noise_psd",
        value=psd,
        extra={"psd": psd, "variance": var, "fs": fs, "n": len(x)},
    )


npsd = noise_psd


def cheatsheet() -> str:
    return "noise_psd({}) -> Noise power spectral density."
