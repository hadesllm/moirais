"""4-bit TurboQuant (7.6x compression)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
from .tqmse import turboquant_mse


def turboquant_4bit(x: np.ndarray) -> DescriptiveResult:
    """4-bit TurboQuant quantization (~7.6x compression).

    :param x: Input vector.
    :return: DescriptiveResult with 4-bit compressed block.
    """
    res = turboquant_mse(x, bits=4)
    res.extra["target_compression"] = 7.6
    return res


def cheatsheet() -> str:
    return "turboquant_4bit(x) -> 4-bit TQ (7.6x compression)"


tq4 = turboquant_4bit
