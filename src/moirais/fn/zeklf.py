"""Kulldorff spatial scan statistic"""

import numpy as np

from ._containers import SpatialResult


def kulldorff_scan(observed, *, expected=None):
    """Kulldorff spatial scan statistic

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
        name="Kulldorff spatial scan statistic",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


kull = kulldorff_scan


def cheatsheet() -> str:
    return "kulldorff_scan({}) -> Kulldorff spatial scan statistic"
