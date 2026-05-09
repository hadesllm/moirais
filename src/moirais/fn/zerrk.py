"""Relative risk kernel ratio"""

import numpy as np

from ._containers import SpatialResult


def relative_risk_kern(observed, *, expected=None):
    """Relative risk kernel ratio

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
        name="Relative risk kernel ratio",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


rela = relative_risk_kern


def cheatsheet() -> str:
    return "relative_risk_kern({}) -> Relative risk kernel ratio"
