# morie.fn — function file (hadesllm/morie)
"""Minimum phase correspondent of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def minimum_phase_correspondent(x: np.ndarray) -> SignalResult:
    """Compute the minimum-phase correspondent via cepstral windowing.

    'Time discovers truth. — Seneca'
    """
    from morie._waveform import minimum_phase_correspondent as _backend

    result = _backend(x)
    return SignalResult(
        name="minimum_phase",
        filtered=result,
        fs=1.0,
        n_samples=len(result),
    )


mnphs = minimum_phase_correspondent


def cheatsheet() -> str:
    return "minimum_phase_correspondent({}) -> Minimum phase correspondent of a signal."
