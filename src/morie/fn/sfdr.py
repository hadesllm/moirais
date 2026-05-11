# morie.fn — function file (hadesllm/morie)
"""Spurious-free dynamic range."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def sfdr_compute(x, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Compute spurious-free dynamic range (SFDR).

    SFDR is the ratio of the fundamental to the largest spurious spectral component.

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
    X = np.fft.rfft(x)
    mag = np.abs(X)
    fund_idx = np.argmax(mag[1:]) + 1
    fund_mag = mag[fund_idx]
    mask = np.ones(len(mag), dtype=bool)
    mask[fund_idx] = False
    mask[0] = False
    if not np.any(mask):
        sfdr_db = float("inf")
    else:
        spur_mag = np.max(mag[mask])
        sfdr_db = 20.0 * np.log10(fund_mag / max(spur_mag, 1e-30))
    return DescriptiveResult(
        name="sfdr",
        value=float(sfdr_db),
        extra={"sfdr_db": float(sfdr_db), "fund_bin": int(fund_idx), "fs": fs},
    )


sfdr = sfdr_compute


def cheatsheet() -> str:
    return "sfdr_compute({}) -> Spurious-free dynamic range."
