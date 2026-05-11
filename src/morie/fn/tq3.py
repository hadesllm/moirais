"""3-bit TurboQuant (10x compression)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .tqmse import turboquant_mse


def turboquant_3bit(x: np.ndarray) -> DescriptiveResult:
    """3-bit TurboQuant quantization (~10x compression).

    :param x: Input vector.
    :return: DescriptiveResult with 3-bit compressed block.
    """
    res = turboquant_mse(x, bits=3)
    res.extra["target_compression"] = 10.0
    return res


def cheatsheet() -> str:
    return "turboquant_3bit(x) -> 3-bit TQ (10x compression)"


tq3 = turboquant_3bit
