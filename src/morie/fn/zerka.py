"""Adaptive bandwidth relative risk"""

import numpy as np

from ._containers import SpatialResult


def risk_adaptive_bw(observed, *, expected=None):
    """Adaptive bandwidth relative risk

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
        name="Adaptive bandwidth relative risk",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


risk = risk_adaptive_bw


def cheatsheet() -> str:
    return "risk_adaptive_bw({}) -> Adaptive bandwidth relative risk"
