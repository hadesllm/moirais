# morie.fn — function file (hadesllm/morie)
"""Parseval's identity verification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The only true wisdom is in knowing you know nothing. — Socrates"


def parseval_identity(x, X_fft=None, **kwargs) -> DescriptiveResult:
    """Verify Parseval's theorem (time-frequency energy equality).

    Sum(|x|^2) == (1/N) * Sum(|X|^2)

    Parameters
    ----------
    x : array-like
        Time-domain signal.
    X_fft : array-like, optional
        FFT of x. Computed if not provided.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if X_fft is None:
        X_fft = np.fft.fft(x)
    else:
        X_fft = np.asarray(X_fft, dtype=complex)
    time_energy = float(np.sum(np.abs(x) ** 2))
    freq_energy = float(np.sum(np.abs(X_fft) ** 2) / len(x))
    ratio = time_energy / (freq_energy + 1e-15)
    holds = bool(np.isclose(time_energy, freq_energy))
    return DescriptiveResult(
        name="parseval_identity",
        value=ratio,
        extra={
            "time_energy": time_energy,
            "freq_energy": freq_energy,
            "ratio": ratio,
            "holds": holds,
        },
    )


prsid = parseval_identity


def cheatsheet() -> str:
    return "parseval_identity({}) -> Parseval's identity verification."
