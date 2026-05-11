# morie.fn — function file (hadesllm/morie)
"""Visitation frequency analysis in custody."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def custody_visits(
    visit_counts: np.ndarray,
    person_ids: np.ndarray | None = None,
) -> DescriptiveResult:
    """Visitation frequency analysis.

    Parameters
    ----------
    visit_counts : ndarray
        Number of visits per person.
    person_ids : ndarray, optional
        Person identifiers (for n unique count).

    Returns
    -------
    DescriptiveResult
    """
    vc = np.asarray(visit_counts, dtype=float)
    n = len(vc)
    pct_zero = float(np.mean(vc == 0))
    return DescriptiveResult(
        name="custody_visits",
        value=float(np.mean(vc)),
        extra={
            "mean": float(np.mean(vc)),
            "median": float(np.median(vc)),
            "std": float(np.std(vc)),
            "pct_zero_visits": pct_zero,
            "n": n,
        },
    )


cstvs = custody_visits


def cheatsheet() -> str:
    return "custody_visits({}) -> Visitation frequency analysis in custody."
