# moirais.fn — function file (hadesllm/moirais)
"""Mean absolute value of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mean_abs_value(x: np.ndarray) -> DescriptiveResult:
    """Compute the mean absolute value (MAV) of a signal.

    'Stay on target.' — Gold Five
    """
    from moirais._waveform import mean_absolute_value as _backend

    result = _backend(x)
    return DescriptiveResult(name="mean_absolute_value", value=float(result))


alias = mean_abs_value


def cheatsheet() -> str:
    return "mean_abs_value({}) -> Mean absolute value of a signal."
