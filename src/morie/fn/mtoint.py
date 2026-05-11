# morie.fn — function file (hadesllm/morie)
"""Intersection safety analysis."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import CrimeResult


def mto_intersection(
    intersection_counts: np.ndarray | list[int],
    volumes: np.ndarray | list[float],
    *,
    per: float = 1e6,
) -> CrimeResult:
    """Compute intersection crash rate per entering vehicles.

    Parameters
    ----------
    intersection_counts : array-like
        Crash counts per intersection.
    volumes : array-like
        Entering vehicle volumes per intersection.
    per : float
        Rate multiplier.

    Returns
    -------
    CrimeResult
    """
    c = np.asarray(intersection_counts, dtype=float)
    v = np.asarray(volumes, dtype=float)
    if len(c) != len(v):
        raise ValueError("counts and volumes must be same length")
    total_crashes = float(c.sum())
    total_vol = float(v.sum())
    rate = total_crashes / total_vol * per if total_vol > 0 else 0.0
    return CrimeResult(
        name="intersection_rate", rate=rate, n=int(total_crashes), extra={"n_intersections": len(c), "per": per}
    )


mtoint = mto_intersection


def cheatsheet() -> str:
    return "mto_intersection({}) -> Intersection safety analysis."
