"""Teager energy operator envelope."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def teager_energy_fn(x: np.ndarray) -> SignalResult:
    """Knowledge itself is power. — Francis Bacon"""
    from moirais._detection import teager_energy as _backend

    result = _backend(x)
    return SignalResult(
        name="teager_energy",
        filtered=result,
        n_samples=int(len(x)),
    )


alias = teager_energy_fn


def cheatsheet() -> str:
    return "teager_energy_fn({}) -> Teager energy operator envelope."
