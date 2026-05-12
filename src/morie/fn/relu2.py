# morie.fn -- function file (hadesllm/morie)
"""ReLU squared activation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def relu_squared(
    x: np.ndarray,
) -> DescriptiveResult:
    r"""Compute ReLU squared activation.

    :math:`\\text{ReLU}^2(x) = (\\max(0, x))^2`

    Used in autoresearch GPT and some modern architectures for sharper gating.

    :param x: Input array.
    :return: DescriptiveResult with activated values in ``extra['output']``.
    """
    x = np.asarray(x, dtype=np.float64)
    output = np.maximum(x, 0.0) ** 2

    return DescriptiveResult(
        name="relu_squared",
        value=float(np.mean(output)),
        extra={"output": output, "sparsity": float(np.mean(x <= 0))},
    )


def cheatsheet() -> str:
    return "relu_squared(x) -> (max(0,x))^2 activation"


relu2 = relu_squared
