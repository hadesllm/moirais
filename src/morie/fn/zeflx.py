"""Flexible spatial scan (Tango)"""

import numpy as np

from ._containers import SpatialResult


def flexible_scan(observed, *, expected=None):
    """Flexible spatial scan (Tango)

    Returns
    -------
    SpatialResult
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float) if expected is not None else np.ones_like(observed) * observed.mean()
    ratio = observed / (expected + 1e-10)
    stat = float(np.max(ratio))
    idx = int(np.argmax(ratio))
    return SpatialResult(
        name="Flexible spatial scan (Tango)",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


flex = flexible_scan


def cheatsheet() -> str:
    return "flexible_scan({}) -> Flexible spatial scan (Tango)"
