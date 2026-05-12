# morie.fn -- function file (hadesllm/morie)
"""Per-facility rate for custody metrics."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def custody_facility_rate(
    counts: np.ndarray,
    populations: np.ndarray,
    facility_ids: np.ndarray,
    *,
    rate_per: int = 1000,
) -> DescriptiveResult:
    """Per-facility rate of incidents, grievances, etc.

    Parameters
    ----------
    counts : ndarray
        Event counts per facility.
    populations : ndarray
        Population per facility.
    facility_ids : ndarray
        Facility identifiers.
    rate_per : int
        Rate denominator.

    Returns
    -------
    DescriptiveResult
    """
    counts = np.asarray(counts, dtype=float)
    populations = np.asarray(populations, dtype=float)
    facility_ids = np.asarray(facility_ids)
    rates = counts / np.maximum(populations, 1e-10) * rate_per
    facility_rates = {str(f): float(r) for f, r in zip(facility_ids, rates)}
    return DescriptiveResult(
        name="custody_facility_rate",
        value=float(np.mean(rates)),
        extra={"facility_rates": facility_rates, "rate_per": rate_per},
    )


cstfr = custody_facility_rate


def cheatsheet() -> str:
    return "custody_facility_rate({}) -> Per-facility rate for custody metrics."
