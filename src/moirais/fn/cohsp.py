# moirais.fn — function file (hadesllm/moirais)
"""Coherence spectrum between two signals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def coherence_spectrum(x, y, fs: float = 1.0, nperseg: int = 256) -> DescriptiveResult:
    """Compute coherence spectrum between *x* and *y*.

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
    from moirais._detection import coherence_spectrum as _cs

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f, Cxy = _cs(x, y, fs=fs, nperseg=nperseg)
    return DescriptiveResult(
        name="coherence_spectrum",
        value=float(np.max(Cxy)),
        extra={"frequencies": f, "coherence": Cxy},
    )


cohsp = coherence_spectrum


def cheatsheet() -> str:
    return "coherence_spectrum({}) -> Coherence spectrum between two signals."
