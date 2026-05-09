# moirais.fn — function file (hadesllm/moirais)
"""Parameter count estimator."""

from __future__ import annotations

from ._containers import DescriptiveResult


def param_count(
    shapes: list[tuple[int, ...]],
    trainable_mask: list[bool] | None = None,
) -> DescriptiveResult:
    """Count total and trainable parameters from layer shapes.

    :param shapes: List of (dim1, dim2, ...) tuples for each parameter tensor.
    :param trainable_mask: Boolean mask; True = trainable. All trainable if None.
    :return: DescriptiveResult with parameter counts.
    """
    import numpy as np

    total = 0
    trainable = 0
    for i, shape in enumerate(shapes):
        n = int(np.prod(shape))
        total += n
        if trainable_mask is None or trainable_mask[i]:
            trainable += n

    frozen = total - trainable

    return DescriptiveResult(
        name="param_count",
        value=total,
        extra={
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "n_tensors": len(shapes),
            "total_millions": total / 1e6,
        },
    )


def cheatsheet() -> str:
    return "param_count(shapes) -> total/trainable parameter count"


prmsz = param_count
