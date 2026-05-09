"""Circular scan statistic"""

import numpy as np

from ._containers import SpatialResult


def scan_circular(observed, *, expected=None):
    """Circular scan statistic

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
        name="Circular scan statistic",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


scan = scan_circular


def cheatsheet() -> str:
    return "scan_circular({}) -> Circular scan statistic"
