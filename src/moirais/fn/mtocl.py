# moirais.fn — function file (hadesllm/moirais)
"""Collision rate per vehicle-kilometres travelled."""

from __future__ import annotations

from moirais.fn._containers import CrimeResult


def mto_collision_rate(
    n_collisions: int,
    vkt: float,
    *,
    per: float = 1e8,
) -> CrimeResult:
    """Compute collision rate per VKT.

    Parameters
    ----------
    n_collisions : int
        Number of collisions.
    vkt : float
        Vehicle-kilometres travelled.
    per : float
        Rate multiplier (default 100 million).

    Returns
    -------
    CrimeResult
    """
    if vkt <= 0:
        raise ValueError("vkt must be positive")
    rate = n_collisions / vkt * per
    return CrimeResult(name="collision_rate", rate=rate, n=n_collisions, extra={"vkt": vkt, "per": per})


mtocl = mto_collision_rate


def cheatsheet() -> str:
    return "mto_collision_rate({}) -> Collision rate per vehicle-kilometres travelled."
