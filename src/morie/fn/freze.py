# morie.fn — function file (hadesllm/morie)
"""Freeze-thaw cycle degradation model. 'Tonight, hell freezes over.' -- Mr. Freeze"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def freeze_thaw(
    n_cycles: int = 100,
    *,
    initial_strength: float = 100.0,
    degradation_rate: float = 0.02,
    threshold: float = 50.0,
    noise_sd: float = 1.0,
    seed: int | None = None,
) -> DescriptiveResult:
    """Model material strength degradation under freeze-thaw cycling.

    Uses an exponential decay model: S(n) = S0 * exp(-r * n) + noise.
    Returns the cycle at which strength drops below the failure threshold.

    Parameters
    ----------
    n_cycles : int
        Number of freeze-thaw cycles to simulate.
    initial_strength : float
        Starting material strength.
    degradation_rate : float
        Exponential decay rate per cycle.
    threshold : float
        Failure threshold.
    noise_sd : float
        Standard deviation of per-cycle noise.
    seed : int or None
        Random seed.

    Returns
    -------
    DescriptiveResult
        ``value`` = cycle at which strength first drops below threshold (or None).
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if initial_strength <= 0:
        raise ValueError("initial_strength must be positive")
    rng = np.random.default_rng(seed)
    cycles = np.arange(n_cycles)
    deterministic = initial_strength * np.exp(-degradation_rate * cycles)
    noise = rng.normal(0, noise_sd, n_cycles)
    strength = deterministic + noise
    below = np.where(strength < threshold)[0]
    failure_cycle = int(below[0]) if len(below) > 0 else None
    half_life = np.log(2) / degradation_rate if degradation_rate > 0 else float("inf")
    return DescriptiveResult(
        name="Freeze-thaw degradation",
        value=failure_cycle,
        extra={
            "n_cycles": n_cycles,
            "initial_strength": initial_strength,
            "degradation_rate": degradation_rate,
            "threshold": threshold,
            "half_life_cycles": float(half_life),
            "final_strength": float(strength[-1]),
            "strength_series": strength.tolist(),
        },
    )


freze = freeze_thaw


def cheatsheet() -> str:
    return "freeze_thaw({}) -> Freeze-thaw cycle degradation model. 'Tonight, hell freezes "
