# morie.fn -- function file (hadesllm/morie)
"""Harmonic analysis (fundamental + overtones).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['hrmns']

_QUOTE = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"


def hrmns(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    n_harmonics: int = 5,
    min_freq: float = 20.0,
    max_freq: float | None = None,
) -> DescriptiveResult:
    """Detect fundamental frequency and harmonics.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to report (including fundamental).
    min_freq : float
        Minimum candidate fundamental frequency.
    max_freq : float or None
        Maximum candidate (default fs/2).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if max_freq is None:
        max_freq = fs / 2.0

    X = np.fft.rfft(x)
    magnitude = np.abs(X)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return DescriptiveResult(
            name="hrmns", value=0.0,
            extra={"fundamental": 0.0, "harmonics": [], "amplitudes": []},
        )

    idx0 = np.where(mask)[0]
    peak_idx = idx0[np.argmax(magnitude[mask])]
    f0 = float(freqs[peak_idx])

    harm_freqs = []
    harm_amps = []
    for h in range(1, n_harmonics + 1):
        fh = f0 * h
        if fh > fs / 2:
            break
        idx = np.argmin(np.abs(freqs - fh))
        harm_freqs.append(float(freqs[idx]))
        harm_amps.append(float(magnitude[idx]))

    return DescriptiveResult(
        name="hrmns",
        value=f0,
        extra={
            "fundamental": f0,
            "harmonics": harm_freqs,
            "amplitudes": harm_amps,
        },
    )


def cheatsheet() -> str:
    return "hrmns({}) -> Harmonic analysis."
