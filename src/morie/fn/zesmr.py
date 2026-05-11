"""Standardized Morbidity Ratio"""

import numpy as np

from ._containers import SpatialResult


def smr_compute(observed, *, expected=None):
    """Standardized Morbidity Ratio

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
        name="Standardized Morbidity Ratio",
        statistic=float(stat) if isinstance(stat, (bool, int, float)) else 0.0,
        extra={},
    )


smr_ = smr_compute


def cheatsheet() -> str:
    return "smr_compute({}) -> Standardized Morbidity Ratio"
