"""Slope sign changes of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def slope_sign_changes_fn(x: np.ndarray, threshold: float = 0.0) -> DescriptiveResult:
    """Count the number of slope sign changes (SSC) in a signal.

    'Wars not make one great.'
    """
    from moirais._waveform import slope_sign_changes as _backend

    result = _backend(x, threshold=threshold)
    return DescriptiveResult(name="slope_sign_changes", value=int(result))


alias = slope_sign_changes_fn


def cheatsheet() -> str:
    return "slope_sign_changes_fn({}) -> Slope sign changes of a signal."
