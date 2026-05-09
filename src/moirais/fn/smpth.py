"""Sampling theorem (Nyquist) check."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "It's a trap!"


def sampling_theorem_check(x, fs: float, fmax: float) -> DescriptiveResult:
    """Check if Nyquist criterion is satisfied: fs >= 2 * fmax.

    Parameters
    ----------
    x : array-like
        Input signal (used to verify length).
    fs : float
        Sampling frequency (Hz).
    fmax : float
        Maximum signal frequency (Hz).

    Returns
    -------
    DescriptiveResult
        value is 1.0 if satisfied, 0.0 otherwise.
    """
    x = np.asarray(x, dtype=float)
    nyquist_rate = 2.0 * fmax
    satisfied = fs >= nyquist_rate
    return DescriptiveResult(
        name="sampling_theorem_check",
        value=1.0 if satisfied else 0.0,
        extra={
            "satisfied": satisfied,
            "fs": fs,
            "fmax": fmax,
            "nyquist_rate": nyquist_rate,
            "n_samples": len(x),
        },
    )


smpth = sampling_theorem_check


def cheatsheet() -> str:
    return "sampling_theorem_check({}) -> Sampling theorem (Nyquist) check."
