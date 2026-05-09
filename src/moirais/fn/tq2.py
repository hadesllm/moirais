"""2-bit TurboQuant (14.6x compression)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .tqmse import turboquant_mse


def turboquant_2bit(x: np.ndarray) -> DescriptiveResult:
    """2-bit TurboQuant quantization (~14.6x compression).

    :param x: Input vector.
    :return: DescriptiveResult with 2-bit compressed block.
    """
    res = turboquant_mse(x, bits=2)
    res.extra["target_compression"] = 14.6
    return res


def cheatsheet() -> str:
    return "turboquant_2bit(x) -> 2-bit TQ (14.6x compression)"


tq2 = turboquant_2bit
