"""Shannon energy envelope of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def shannon_energy_fn(x: np.ndarray) -> SignalResult:
    """Compute the Shannon energy envelope.

    'Size matters not.'
    """
    from morie._detection import shannon_energy as _backend

    result = _backend(x)
    return SignalResult(
        name="shannon_energy",
        filtered=result,
        n_samples=int(len(x)),
    )


alias = shannon_energy_fn


def cheatsheet() -> str:
    return "shannon_energy_fn({}) -> Shannon energy envelope of a signal."
