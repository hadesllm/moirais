"""Patrol efficiency metrics."""

from __future__ import annotations

from moirais.fn._containers import ESRes


def tps_patrol_efficiency(
    crimes: int,
    patrol_hours: float,
) -> ESRes:
    """Compute crimes per patrol-hour efficiency ratio.

    Parameters
    ----------
    crimes : int
        Number of crimes in area/period.
    patrol_hours : float
        Total patrol hours deployed.

    Returns
    -------
    ESRes
    """
    if patrol_hours <= 0:
        raise ValueError("patrol_hours must be positive")
    ratio = crimes / patrol_hours
    return ESRes(
        measure="crimes_per_patrol_hour",
        estimate=ratio,
        n=crimes,
        extra={"patrol_hours": patrol_hours},
    )


tpspr = tps_patrol_efficiency


def cheatsheet() -> str:
    return "tps_patrol_efficiency({}) -> Patrol efficiency metrics."
